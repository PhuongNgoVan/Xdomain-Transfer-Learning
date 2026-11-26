"""Native training, dev selection, atomic recovery and Task 10 extension hooks."""
import copy
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from src.common.seed import set_seed
from src.phase2.models import count_model_parameters
from src.phase2.protocol import canonical_digest, compute_dev_selection_score


def build_default_adamw_optimizer(model, settings):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if name.endswith("bias") or "layernorm" in name.lower()
             or "layer_norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": settings["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=settings["lr"])


class Phase2Trainer:
    """fit accepts only train/dev; predict uses FP32 after restoring best weights."""

    def __init__(self, model, tokenizer, target_domain, num_labels, seed, settings, *,
                 optimizer_builder=None, checkpoint_saver=None):
        self.settings = {"batch_size": 16, "gradient_accumulation_steps": 2, "epochs": 10,
                         "evals_per_epoch": 1, "patience": 2, "warmup_ratio": .1,
                         "weight_decay": .01, "max_grad_norm": 1., "max_length": 256,
                         "precision": "bf16", "strict_precision": True, **settings}
        self.seed, self.num_labels, self.target_domain = int(seed), num_labels, target_domain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = self.settings["precision"]
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        supported = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        if precision == "bf16" and self.settings["strict_precision"] and not supported:
            raise RuntimeError("strict bf16 requires a CUDA device with bf16 support")
        self.amp = precision == "bf16" and supported
        self.model, self.tokenizer = model.to(self.device).float(), tokenizer
        self.optimizer_builder = optimizer_builder or build_default_adamw_optimizer
        self.checkpoint_saver = checkpoint_saver
        self.history = []

    def _encode(self, texts):
        encoded = self.tokenizer(list(texts), padding=False, truncation=True,
                                 max_length=self.settings["max_length"])
        return [{k: v[i] for k, v in encoded.items()} for i in range(len(texts))]

    def _batch(self, encodings, labels=None):
        batch = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
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
                ids = order[start:start + self.settings["batch_size"]]
                logits = self.model(**self._batch([encoded[i] for i in ids])).logits
                result[ids] = logits.float().softmax(-1).cpu().numpy()
        return result

    def _save_resume(self, state, path):
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def fit(self, train_texts, train_labels, dev_texts, dev_labels, work_dir):
        s = self.settings
        for key in ("batch_size", "gradient_accumulation_steps", "epochs", "evals_per_epoch", "patience"):
            if int(s[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if not train_texts or not dev_texts or len(train_texts) != len(train_labels) or len(dev_texts) != len(dev_labels):
            raise ValueError("Nonempty matching train/dev texts and labels required")
        identity = canonical_digest({"settings": s, "seed": self.seed, "target": self.target_domain,
            "num_labels": self.num_labels, "train": list(train_texts), "labels": list(train_labels),
            "dev": list(dev_texts), "dev_labels": list(dev_labels)})
        encoded = self._encode(train_texts)
        full = self.tokenizer(list(train_texts), padding=False, truncation=False)["input_ids"]
        self.trunc_rate = sum(len(x) > s["max_length"] for x in full) / len(full)
        n_batches = math.ceil(len(encoded) / s["batch_size"])
        steps_epoch = math.ceil(n_batches / s["gradient_accumulation_steps"])
        total_steps = steps_epoch * s["epochs"]
        # Evaluation boundaries within each epoch, including its final update.
        eval_steps = {math.ceil(i * steps_epoch / s["evals_per_epoch"])
                      for i in range(1, s["evals_per_epoch"] + 1)}
        self.optimizer = self.optimizer_builder(self.model, s)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, math.ceil(total_steps * s["warmup_ratio"]), total_steps)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        state_file = work_dir / "resume.pt"
        epoch = cursor = step = patience = 0
        best_score, best_step, best_epoch, best_state = -1., 0, 0., None
        loss_sum = loss_items = 0
        elapsed = 0.
        self.history = []
        set_seed(self.seed)
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            if state.get("identity") != identity:
                raise ValueError("Resume train/dev/settings identity mismatch")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            epoch, cursor, step, patience = [state[k] for k in ("epoch", "cursor", "step", "patience")]
            best_score, best_step, best_epoch, best_state = [state[k] for k in ("best_score", "best_step", "best_epoch", "best_state")]
            loss_sum, loss_items, elapsed = [state[k] for k in ("loss_sum", "loss_items", "train_seconds")]
            self.history = state["history"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def save(next_epoch, next_cursor):
            state = dict(identity=identity, model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                epoch=next_epoch, cursor=next_cursor, step=step, patience=patience,
                best_score=best_score, best_step=best_step, best_epoch=best_epoch, best_state=best_state,
                history=self.history, loss_sum=loss_sum, loss_items=loss_items,
                train_seconds=elapsed + time.perf_counter() - started,
                torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
            self._save_resume(state, state_file)

        stop = patience >= s["patience"]
        while epoch < s["epochs"] and not stop:
            ids = np.random.default_rng(self.seed + epoch).permutation(len(encoded)).tolist()
            for start in range(0, len(ids), s["batch_size"] * 50):
                ids[start:start + s["batch_size"] * 50] = sorted(ids[start:start + s["batch_size"] * 50],
                    key=lambda i: len(encoded[i]["input_ids"]), reverse=True)
            batches = [ids[i:i + s["batch_size"]] for i in range(0, len(ids), s["batch_size"])]
            while cursor < n_batches:
                chunk = batches[cursor:cursor + s["gradient_accumulation_steps"]]
                n_items = sum(map(len, chunk))
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                for batch_ids in chunk:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                        loss = self.model(**self._batch([encoded[i] for i in batch_ids],
                            [train_labels[i] for i in batch_ids])).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss_sum += loss.item() * len(batch_ids)
                    loss_items += len(batch_ids)
                    (loss * (len(batch_ids) / n_items)).backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], s["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()
                cursor += len(chunk)
                step += 1
                if math.ceil(cursor / s["gradient_accumulation_steps"]) in eval_steps:
                    score = compute_dev_selection_score(dev_labels, self.predict(dev_texts).argmax(1), self.target_domain)
                    position = epoch + cursor / n_batches
                    self.history.append(dict(epoch=position, step=step, train_loss=loss_sum / loss_items,
                                             dev_selection_score=score))
                    loss_sum = loss_items = 0
                    if score > best_score:
                        best_score, best_step, best_epoch = score, step, position
                        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                        patience = 0
                    else:
                        patience += 1
                    stop = patience >= s["patience"]
                # Every completed update is recoverable, including between evaluations.
                save(epoch + 1 if cursor == n_batches else epoch, 0 if cursor == n_batches else cursor)
                if stop:
                    break
            epoch += 1
            cursor = 0
        if best_state is None:
            raise RuntimeError("No best model selected on dev")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = elapsed + time.perf_counter() - started
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.cost = {**count_model_parameters(self.model), "train_seconds": seconds,
            "gpu_hours": seconds / 3600 if self.device.type == "cuda" else 0.,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == "cuda" else 0.,
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "n_gpus": int(self.device.type == "cuda"), "actual_update_steps": step,
            "trunc_rate": self.trunc_rate, "effective_precision": "bf16" if self.amp else "fp32"}
        return dict(best_score=best_score, dev_selection_score=best_score, best_step=best_step,
                    best_epoch=best_epoch, early_stopped=stop, actual_update_steps=step,
                    history=self.history, trunc_rate=self.trunc_rate, cost=self.cost)

    def save(self, path):
        if self.checkpoint_saver is not None:
            return self.checkpoint_saver(self.model, self.tokenizer, Path(path))
        return self.model.save_stage1_checkpoint(self.tokenizer, path)
