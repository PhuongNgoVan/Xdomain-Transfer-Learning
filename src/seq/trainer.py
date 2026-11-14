"""Native PyTorch training with explicit AMP and resumable best-model state."""
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.common.seed import set_seed
from src.plm.model_ckpt import Stage1Classifier


def load_encoder_new_head(checkpoint, num_labels, head_seed, dropout=0.1):
    encoder_dir = Path(checkpoint) / "encoder"
    encoder = AutoModel.from_pretrained(encoder_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    set_seed(int(head_seed))
    return Stage1Classifier(encoder, num_labels=num_labels, dropout=dropout), tokenizer


class TorchTrainer:
    """Test data is not accepted by fit; predict is called only after best restore."""

    def __init__(self, checkpoint, num_labels, seed, settings):
        self.settings = settings
        self.seed = int(seed)
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = settings["precision"]
        if precision not in ("bf16", "fp32"):
            raise ValueError("seq-v1 supports explicit bf16 or fp32; no silent fallback")
        if precision == "bf16" and settings.get("strict_precision", True):
            if self.device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise RuntimeError("strict bf16 requires a CUDA device with bf16 support")
        self.amp = precision == "bf16"
        print(f"[Precision] {precision} (autocast={self.amp}, device={self.device.type})")
        self.model, self.tokenizer = load_encoder_new_head(checkpoint, num_labels, seed, settings["dropout"])
        self.model.to(self.device)
        self.history = []

    def _encode(self, texts):
        encoded = self.tokenizer(texts, padding=False, truncation=True, max_length=self.settings["max_length"])
        return [{key: value[i] for key, value in encoded.items()} for i in range(len(texts))]

    def _batch(self, encodings, labels=None):
        batch = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long, device=self.device)
        return batch

    def predict(self, texts):
        encoded = self._encode(texts)
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]["input_ids"]), reverse=True)
        result = np.empty((len(texts), self.num_labels), dtype=np.float64)
        self.model.float().eval()
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
            for start in range(0, len(order), self.settings["batch_size"]):
                indices = order[start:start + self.settings["batch_size"]]
                logits = self.model(**self._batch([encoded[i] for i in indices])).logits
                result[indices] = logits.float().softmax(-1).cpu().numpy()
        return result

    def fit(self, train_texts, train_labels, dev_texts, dev_labels, work_dir):
        s = self.settings
        encoded = self._encode(train_texts)
        full = self.tokenizer(train_texts, padding=False, truncation=False)["input_ids"]
        self.trunc_rate = sum(len(x) > s["max_length"] for x in full) / len(full)
        n_batches = math.ceil(len(encoded) / s["batch_size"])
        steps_epoch = math.ceil(n_batches / s["gradient_accumulation_steps"])
        total_steps = steps_epoch * s["epochs"]
        cadence = max(1, steps_epoch // s["evals_per_epoch"])
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            (no_decay if name.endswith("bias") or "layernorm" in name.lower() else decay).append(parameter)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": s["weight_decay"]},
                                      {"params": no_decay, "weight_decay": 0.0}], lr=s["lr"])
        scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * s["warmup_ratio"]), total_steps)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        state_file = work_dir / "resume.pt"
        epoch, cursor, step, patience = 0, 0, 0, 0
        best_score, best_step, best_epoch, best_state = -1.0, 0, 0.0, None
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            epoch, cursor, step, patience = [state[k] for k in ("epoch", "cursor", "step", "patience")]
            best_score, best_step, best_epoch, best_state = [state[k] for k in ("best_score", "best_step", "best_epoch", "best_state")]
            self.history = state["history"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        def save(next_epoch, next_cursor):
            state = dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         epoch=next_epoch, cursor=next_cursor, step=step, patience=patience,
                         best_score=best_score, best_step=best_step, best_epoch=best_epoch, best_state=best_state,
                         history=self.history, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                         python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
            tmp = state_file.with_suffix(".tmp")
            torch.save(state, tmp)
            tmp.replace(state_file)
        stop = patience >= s["patience"]
        while epoch < s["epochs"] and not stop:
            rng = np.random.default_rng(self.seed + epoch)
            indices = rng.permutation(len(encoded)).tolist()
            # Deterministic length-grouped mega-batches, with epoch-specific ordering.
            for start in range(0, len(indices), s["batch_size"] * 50):
                indices[start:start + s["batch_size"] * 50] = sorted(indices[start:start + s["batch_size"] * 50], key=lambda i: len(encoded[i]["input_ids"]), reverse=True)
            batches = [indices[start:start + s["batch_size"]] for start in range(0, len(indices), s["batch_size"])]
            while cursor < len(batches):
                chunk = batches[cursor:cursor + s["gradient_accumulation_steps"]]
                n_items = sum(map(len, chunk))
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                for ids in chunk:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                        loss = self.model(**self._batch([encoded[i] for i in ids], [train_labels[i] for i in ids])).loss
                        loss = loss * (len(ids) / n_items)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss; run is not complete")
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), s["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                step += 1
                cursor += len(chunk)
                if step % cadence == 0 or cursor == len(batches):
                    score = f1_score(dev_labels, self.predict(dev_texts).argmax(1), average="macro", labels=list(range(self.num_labels)), zero_division=0)
                    position = epoch + cursor / len(batches)
                    self.history.append({"step": step, "epoch": position, "dev_macro_f1": float(score)})
                    if score > best_score:
                        best_score, best_step, best_epoch = float(score), step, position
                        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                        patience = 0
                    else:
                        patience += 1
                    save(epoch + 1 if cursor == len(batches) else epoch, 0 if cursor == len(batches) else cursor)
                    if patience >= s["patience"]:
                        stop = True
                        break
            epoch += 1
            cursor = 0
        if best_state is None:
            raise RuntimeError("No best model was selected on dev")
        self.model.load_state_dict(best_state)
        self.model.eval()
        return {"best_step": best_step, "best_epoch": best_epoch, "history": self.history, "trunc_rate": self.trunc_rate}

    def save(self, path):
        self.model.save_stage1_checkpoint(self.tokenizer, path)
