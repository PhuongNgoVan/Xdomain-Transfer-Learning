"""Task 11 LLM zero-shot and few-shot evaluation on Phase 2 10% splits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import yaml

from src.common.constants import EMOTIONS7, LABELS3
from src.common.hashing import sha256_file
from src.data.preprocess import preprocess_text
from src.phase2.audit_task01_02 import check_underscore_leakage
from src.phase2.protocol import canonical_digest, compute_dev_selection_score, load_phase2_config
from src.phase2.run import atomic_json, score_view
from src.phase2.subsets import load_phase2_target_splits
from src.zs.infer import _atomic_csv, _git_commit, _prediction_frame
from src.zs.mapping import validate_probabilities

STRATEGIES: Tuple[str, ...] = ("zero_shot", "stratified_fixed", "tfidf_knn", "balanced_tfidf_knn")
FEW_SHOT_STRATEGIES: Tuple[str, ...] = ("stratified_fixed", "tfidf_knn", "balanced_tfidf_knn")
PROMPT_STYLES: Tuple[str, ...] = ("concise", "rubric")

DOMAIN_LABELS: Dict[str, Tuple[str, ...]] = {
    "vsmec": EMOTIONS7,
    "neuesc": LABELS3,
}

FALLBACK_LABEL_ID: Dict[str, int] = {
    "vsmec": 6,   # Other
    "neuesc": 1,  # neutral
}

_LABEL_RUBRICS: Dict[str, str] = {
    "vsmec": (
        "- Enjoyment: vui vẻ, thích thú, hài lòng, yêu thích, khen ngợi.\n"
        "- Sadness: buồn bã, thất vọng, tủi thân, tiếc nuối.\n"
        "- Fear: sợ hãi, lo lắng, bất an, hoảng hốt.\n"
        "- Anger: tức giận, bức xúc, phẫn nộ, cáu gắt.\n"
        "- Disgust: ghê tởm, chán ghét, khinh bỉ, phản cảm.\n"
        "- Surprise: ngạc nhiên, bất ngờ, sửng sốt.\n"
        "- Other: trung tính hoặc cảm xúc không rõ vào 6 nhóm trên."
    ),
    "neuesc": (
        "- negative: thái độ tiêu cực, chê bai, phàn nàn, bất mãn, công kích hoặc độc hại.\n"
        "- neutral: trung tính, khách quan, cung cấp thông tin hoặc không rõ khen/chê.\n"
        "- positive: thái độ tích cực, khen ngợi, đồng tình, hài lòng, khích lệ."
    ),
}

PRETRAINING_LEAKAGE_NOTE = (
    "Public Vietnamese datasets (UIT-VSMEC, NEU-ESC) may overlap with pretraining "
    "corpora of open-weight LLMs; results are an upper-bound reference rather than "
    "strictly unseen zero-shot generalization."
)


def load_llm_task_config(path: str | Path = "configs/task11_llm.yaml") -> Dict[str, Any]:
    """Load Task 11 LLM config and validate locked phase2-v1 invariants."""
    cfg_path = Path(path)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    proto_ref = doc.get("protocol_config", "phase2.yaml")
    proto_path = (cfg_path.resolve().parent / proto_ref).resolve()
    proto = load_phase2_config(proto_path)
    llm_proto = proto["scope_and_budget"]["llm_reference_10pct"]
    run_cfg = dict(doc.get("llm", {}))
    if float(run_cfg.get("temperature", llm_proto["temperature"])) != 0.0:
        raise ValueError("Task 11 LLM requires deterministic greedy decoding (temperature=0.0)")
    if run_cfg.get("prompt_selection_split", llm_proto["prompt_selection_split"]) != "dev_only":
        raise ValueError("Task 11 LLM prompt selection must be dev_only")
    if run_cfg.get("few_shot_pool", llm_proto["few_shot_pool"]) != "train_10pct_only":
        raise ValueError("Task 11 LLM few-shot pool must be train_10pct_only")
    return {"protocol_cfg": proto, "protocol_path": str(proto_path), "llm": run_cfg}


def _clean_texts(frame: pd.DataFrame, split: str) -> List[str]:
    raw = frame["text"].astype(str).tolist()
    texts = [preprocess_text(t, emoji_mode="keep") for t in raw]
    if not texts or any(not t.strip() for t in texts):
        raise ValueError(f"Empty {split} input text")
    if check_underscore_leakage(raw, texts)["status"] != "PASS":
        raise ValueError(f"Underscore leakage in {split}")
    return texts


def load_llm_splits(
    domain: str,
    *,
    fraction: int = 10,
    seed: int = 0,
    clean_dir: str | Path = "data/clean",
    subsets_dir: str | Path = "data/subsets_phase2",
    include_test: bool = False,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Load train (few-shot pool), dev, and optionally test from frozen Phase 2 splits."""
    if domain not in DOMAIN_LABELS:
        raise ValueError(f"Unsupported domain: {domain}")
    if int(fraction) != 10:
        raise ValueError("Task 11 LLM reference is defined at fraction=10 only")
    train_df, dev_df, test_df, meta = load_phase2_target_splits(
        domain,
        fraction=fraction,
        seed=seed,
        clean_dir=clean_dir,
        subsets_dir=subsets_dir,
        include_test=include_test,
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError("max_examples must be positive")
        dev_df = dev_df.iloc[: int(max_examples)].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df.iloc[: int(max_examples)].reset_index(drop=True)
    elif test_df is not None:
        test_df = test_df.reset_index(drop=True)

    train_texts = _clean_texts(train_df, "train")
    dev_texts = _clean_texts(dev_df, "dev")
    test_texts = _clean_texts(test_df, "test") if test_df is not None else None
    return {
        "domain": domain,
        "labels": DOMAIN_LABELS[domain],
        "train_df": train_df,
        "train_texts": train_texts,
        "train_labels": train_df["label"].astype(int).tolist(),
        "dev_df": dev_df,
        "dev_texts": dev_texts,
        "dev_labels": dev_df["label"].astype(int).tolist(),
        "test_df": test_df,
        "test_texts": test_texts,
        "test_labels": test_df["label"].astype(int).tolist() if test_df is not None else None,
        "provenance": {
            **meta,
            "max_examples": max_examples,
            "include_test": include_test,
            "data_hash": canonical_digest(meta["source_files_sha256"]),
        },
    }


class FewShotSelector:
    """Deterministic selector drawing examples strictly from the 10% train pool."""

    def __init__(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        label_names: Sequence[str],
        *,
        seed: int = 0,
        shots_per_class: int = 1,
    ) -> None:
        self.train_texts = list(train_texts)
        self.train_labels = [int(y) for y in train_labels]
        self.label_names = tuple(label_names)
        self.seed = int(seed)
        self.shots_per_class = int(shots_per_class)
        self.num_classes = len(self.label_names)
        self.total_shots = self.shots_per_class * self.num_classes

        self._by_class: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for idx, lbl in enumerate(self.train_labels):
            self._by_class[lbl].append(idx)

        rng = np.random.default_rng(20261127 + self.seed)
        fixed: List[int] = []
        for cls_id in range(self.num_classes):
            pool = self._by_class[cls_id]
            if not pool:
                raise ValueError(f"Empty class {cls_id} in train 10% pool")
            take = min(self.shots_per_class, len(pool))
            chosen = rng.choice(pool, size=take, replace=False).tolist()
            fixed.extend(chosen)
        self._fixed_indices = fixed

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True
        )
        self._train_matrix = self._vectorizer.fit_transform(self.train_texts)

    def select_batch(self, query_texts: Sequence[str], strategy: str) -> List[List[Tuple[str, str]]]:
        """Return few-shot `(text, label_name)` pairs per query for `strategy`."""
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        if strategy == "zero_shot":
            return [[] for _ in query_texts]
        if strategy == "stratified_fixed":
            ex = [(self.train_texts[i], self.label_names[self.train_labels[i]]) for i in self._fixed_indices]
            return [list(ex) for _ in query_texts]

        q_mat = self._vectorizer.transform(list(query_texts))
        sims = linear_kernel(q_mat, self._train_matrix)
        results: List[List[Tuple[str, str]]] = []
        for row in sims:
            if strategy == "tfidf_knn":
                order = np.argsort(row)[-self.total_shots :]
                chosen = order.tolist()
            elif strategy == "balanced_tfidf_knn":
                picked: List[Tuple[float, int]] = []
                for cls_id in range(self.num_classes):
                    pool = np.asarray(self._by_class[cls_id], dtype=np.int64)
                    cls_scores = row[pool]
                    take = min(self.shots_per_class, len(pool))
                    top_local = np.argsort(cls_scores)[-take:]
                    for local_idx in top_local:
                        idx = int(pool[local_idx])
                        picked.append((float(row[idx]), idx))
                picked.sort(key=lambda item: item[0])
                chosen = [idx for _, idx in picked]
            else:  # pragma: no cover - guarded above
                raise ValueError(strategy)
            results.append(
                [(self.train_texts[i], self.label_names[self.train_labels[i]]) for i in chosen]
            )
        return results


def build_llm_messages(
    query_text: str,
    domain: str,
    examples: Sequence[Tuple[str, str]],
    prompt_style: str = "concise",
) -> List[Dict[str, str]]:
    """Construct chat messages for Vietnamese emotion/sentiment classification."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    labels = DOMAIN_LABELS[domain]
    label_list = ", ".join(labels)
    task_desc = (
        "Phân loại cảm xúc câu bình luận tiếng Việt vào đúng 1 trong 7 nhãn"
        if domain == "vsmec"
        else "Phân loại quan điểm/cực tính câu tiếng Việt vào đúng 1 trong 3 nhãn"
    )
    system_parts = [
        f"Bạn là hệ thống phân loại văn bản tiếng Việt. {task_desc}: {label_list}."
    ]
    if prompt_style == "rubric":
        system_parts.append("Định nghĩa nhãn:\n" + _LABEL_RUBRICS[domain])
    system_parts.append(
        f"Quy tắc trả lời: CHỈ xuất đúng 1 tên nhãn hợp lệ trong danh sách [{label_list}], "
        "không giải thích, không thêm ký tự khác."
    )
    user_lines: List[str] = []
    if examples:
        user_lines.append("Ví dụ minh họa:")
        for i, (ex_text, ex_label) in enumerate(examples, 1):
            user_lines.append(f"Ví dụ {i}:\nVăn bản: {ex_text}\nNhãn: {ex_label}")
        user_lines.append("")
    user_lines.append(f"Văn bản cần phân loại:\n{query_text}\nNhãn:")
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def render_prompt_text(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Apply chat template when available; otherwise format plain prompt."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages) + "\n[ASSISTANT]\n"


def parse_llm_label(raw_output: str, domain: str) -> Tuple[int, str, bool]:
    """Parse generated text into target label ID, canonical name, and validity flag."""
    labels = DOMAIN_LABELS[domain]
    lower_map = {name.lower(): (idx, name) for idx, name in enumerate(labels)}
    text = str(raw_output or "").strip()
    if not text:
        fb = FALLBACK_LABEL_ID[domain]
        return fb, labels[fb], False

    first_line = text.splitlines()[0].strip()
    cleaned = re.sub(r'^[\s"\'`*:\-\[\]().,;]+|[\s"\'`*:\-\[\]().,;]+$', "", first_line).lower()
    if cleaned in lower_map:
        idx, name = lower_map[cleaned]
        return idx, name, True

    # Try JSON or "Nhãn: <label>" / "Label: <label>" patterns
    json_match = re.search(r'"?(?:label|nhãn)"?\s*:\s*"?([a-zA-Z]+)"?', text, flags=re.IGNORECASE)
    if json_match:
        cand = json_match.group(1).lower()
        if cand in lower_map:
            idx, name = lower_map[cand]
            return idx, name, True

    # Boundary word match across allowed labels (prefer earliest single occurrence)
    hits: List[Tuple[int, int, str]] = []
    for idx, name in enumerate(labels):
        m = re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE)
        if m:
            hits.append((m.start(), idx, name))
    if hits:
        hits.sort(key=lambda item: item[0])
        _, idx, name = hits[0]
        return idx, name, True

    fb = FALLBACK_LABEL_ID[domain]
    return fb, labels[fb], False


def verify_hf_model_metadata(model_name: str, verify_license: bool = True) -> Dict[str, Any]:
    """Fetch model revision and license from Hugging Face Hub metadata."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        card = getattr(info, "card_data", None)
        license_name = getattr(card, "license", None) if card is not None else None
        if not license_name and getattr(info, "tags", None):
            for tag in info.tags:
                if str(tag).startswith("license:"):
                    license_name = str(tag).split(":", 1)[1]
                    break
        if verify_license and not license_name:
            raise ValueError(f"Could not verify license for {model_name}; check model card before running.")
        return {
            "model_name": model_name,
            "model_revision": getattr(info, "sha", "unknown"),
            "license": license_name or "unspecified",
        }
    except Exception as exc:
        if verify_license:
            raise RuntimeError(f"License/version verification failed for {model_name}: {exc}") from exc
        return {"model_name": model_name, "model_revision": "unknown", "license": "unverified"}


class HuggingFaceLLMClassifier:
    """Greedy-decoding zero/few-shot classifier for open-weight CausalLMs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        precision: str = "bf16",
        max_input_length: int = 1024,
        max_new_tokens: int = 16,
        batch_size: int = 16,
        verify_license: bool = True,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if precision not in ("bf16", "fp16", "fp32", "4bit", "8bit"):
            raise ValueError(f"Unsupported LLM precision: {precision}")
        self.model_name = model_name
        self.precision = precision
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.hub_meta = verify_hf_model_metadata(model_name, verify_license=verify_license and not local_files_only)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"local_files_only": local_files_only, "device_map": "auto"}
        if precision == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
        elif precision == "fp32":
            load_kwargs["torch_dtype"] = torch.float32
        elif precision in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                if precision == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
        if not torch.cuda.is_available():
            load_kwargs.pop("device_map", None)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gen_cfg, attr):
                    setattr(gen_cfg, attr, None)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = 0

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        import torch

        device = next(self.model.parameters()).device
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        restored = [""] * len(prompts)
        total_batches = max(1, math.ceil(len(order) / self.batch_size))
        with torch.inference_mode():
            for batch_idx, start in enumerate(range(0, len(order), self.batch_size), 1):
                idxs = order[start : start + self.batch_size]
                batch_prompts = [prompts[i] for i in idxs]
                enc = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                ).to(device)
                prompt_len = enc["input_ids"].shape[1]
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                decoded = self.tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
                for pos, text in zip(idxs, decoded):
                    restored[pos] = text.strip()
                if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                    print(f"[llm] batch {batch_idx}/{total_batches} ({min(start + self.batch_size, len(order))}/{len(order)})", flush=True)
        return restored

    def metadata(self) -> Dict[str, Any]:
        import torch

        return {
            **self.hub_meta,
            "precision": self.precision,
            "quantization": self.precision if self.precision in ("4bit", "8bit") else "none",
            "temperature": 0.0,
            "do_sample": False,
            "total_params": int(self.total_params),
            "trainable_params": int(self.trainable_params),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }


def _one_hot_probabilities(pred_ids: Sequence[int], num_labels: int) -> np.ndarray:
    probs = np.zeros((len(pred_ids), num_labels), dtype=np.float64)
    for row_idx, cls_id in enumerate(pred_ids):
        probs[row_idx, int(cls_id)] = 1.0
    return validate_probabilities(probs, num_labels)


def evaluate_llm_split(
    split_name: str,
    df: pd.DataFrame,
    texts: Sequence[str],
    labels: Sequence[int],
    domain: str,
    strategy: str,
    prompt_style: str,
    selector: FewShotSelector,
    predictor_fn: Callable[[Sequence[str]], Sequence[str]],
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Run deterministic LLM prediction and score all domain views on one split."""
    examples_per_row = selector.select_batch(texts, strategy)
    prompts = [
        render_prompt_text(tokenizer, build_llm_messages(text, domain, exs, prompt_style=prompt_style))
        for text, exs in zip(texts, examples_per_row)
    ]
    t0 = time.perf_counter()
    raw_outputs = list(predictor_fn(prompts))
    elapsed = time.perf_counter() - t0
    if len(raw_outputs) != len(texts):
        raise ValueError(f"Expected {len(texts)} LLM outputs, got {len(raw_outputs)}")

    parsed_ids: List[int] = []
    parsed_names: List[str] = []
    valid_flags: List[bool] = []
    for out in raw_outputs:
        lbl_id, lbl_name, valid = parse_llm_label(out, domain)
        parsed_ids.append(lbl_id)
        parsed_names.append(lbl_name)
        valid_flags.append(valid)

    num_labels = len(DOMAIN_LABELS[domain])
    probs = _one_hot_probabilities(parsed_ids, num_labels)
    pred_frame = _prediction_frame(df, domain, num_labels, probs)
    dev_score = compute_dev_selection_score(
        labels, pred_frame["y_pred7" if num_labels == 7 else "y_pred3_sum"].to_numpy(), domain
    )
    views = (
        [("V3-NEU", "none")]
        if domain == "neuesc"
        else [("V7-VSMEC", "none"), ("V5-VSMEC", "none"), ("V3-VSMEC", "sum"), ("V3-VSMEC-strict", "sum")]
    )
    view_metrics = {f"{v}/{m}": score_view(pred_frame, v, m) for v, m in views}
    raw_df = pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "y_true": list(labels),
            "y_pred": parsed_ids,
            "pred_label": parsed_names,
            "valid_parse": valid_flags,
            "raw_output": raw_outputs,
        }
    )
    invalid_count = sum(not v for v in valid_flags)
    return {
        "split": split_name,
        "domain": domain,
        "strategy": strategy,
        "prompt_style": prompt_style,
        "n_eval": len(texts),
        "dev_selection_score": float(dev_score),
        "view_metrics": view_metrics,
        "invalid_output_count": int(invalid_count),
        "invalid_output_rate": float(invalid_count / len(texts)),
        "elapsed_seconds": float(elapsed),
        "throughput_samples_per_sec": float(len(texts) / elapsed) if elapsed > 0 else 0.0,
        "pred_frame": pred_frame,
        "raw_frame": raw_df,
    }


def run_llm_dev_sweep(
    domain: str,
    predictor_fn: Callable[[Sequence[str]], Sequence[str]],
    *,
    tokenizer: Any = None,
    model_meta: Optional[Dict[str, Any]] = None,
    strategies: Sequence[str] = STRATEGIES,
    prompt_styles: Sequence[str] = PROMPT_STYLES,
    shots_per_class: int = 1,
    seed: int = 0,
    clean_dir: str | Path = "data/clean",
    subsets_dir: str | Path = "data/subsets_phase2",
    out_dir: str | Path = "results/phase2/llm",
    max_examples: Optional[int] = None,
    config_path: str | Path = "configs/task11_llm.yaml",
) -> Dict[str, Any]:
    """Evaluate zero-shot and few-shot strategies on dev ONLY and lock selection JSON."""
    cfg_bundle = load_llm_task_config(config_path)
    data = load_llm_splits(
        domain,
        fraction=10,
        seed=seed,
        clean_dir=clean_dir,
        subsets_dir=subsets_dir,
        include_test=False,
        max_examples=max_examples,
    )
    assert data["test_df"] is None and data["test_texts"] is None
    selector = FewShotSelector(
        data["train_texts"],
        data["train_labels"],
        data["labels"],
        seed=seed,
        shots_per_class=shots_per_class,
    )
    root = Path(out_dir) / domain
    root.mkdir(parents=True, exist_ok=True)
    primary_view = "V3-NEU/none" if domain == "neuesc" else "V5-VSMEC/none"

    rows: List[Dict[str, Any]] = []
    for style in prompt_styles:
        for strategy in strategies:
            res = evaluate_llm_split(
                "dev",
                data["dev_df"],
                data["dev_texts"],
                data["dev_labels"],
                domain,
                strategy,
                style,
                selector,
                predictor_fn,
                tokenizer=tokenizer,
            )
            tag = f"{strategy}__{style}"
            _atomic_csv(res["pred_frame"], root / f"dev_pred_{tag}.csv")
            _atomic_csv(res["raw_frame"], root / f"dev_raw_{tag}.csv")
            prim = res["view_metrics"][primary_view]
            rows.append(
                {
                    "domain": domain,
                    "strategy": strategy,
                    "prompt_style": style,
                    "is_few_shot": strategy != "zero_shot",
                    "shots": 0 if strategy == "zero_shot" else selector.total_shots,
                    "n_eval": res["n_eval"],
                    "dev_selection_score": res["dev_selection_score"],
                    "macro_f1": prim["macro_f1"],
                    "weighted_f1": prim["weighted_f1"],
                    "accuracy": prim["accuracy"],
                    "invalid_output_count": res["invalid_output_count"],
                    "invalid_output_rate": res["invalid_output_rate"],
                    "elapsed_seconds": res["elapsed_seconds"],
                    "data_hash": data["provenance"]["data_hash"],
                    "ids_sha256": data["provenance"]["ids_sha256"],
                    "max_examples": max_examples,
                }
            )
    sweep_df = pd.DataFrame(rows)
    _atomic_csv(sweep_df, root / f"dev_sweep_{domain}.csv")

    zs_df = sweep_df.loc[sweep_df["strategy"].eq("zero_shot")]
    fs_df = sweep_df.loc[sweep_df["strategy"].ne("zero_shot")]
    selection: Dict[str, Any] = {
        "protocol_version": cfg_bundle["protocol_cfg"]["protocol_version"],
        "protocol_hash": canonical_digest(cfg_bundle["protocol_cfg"]),
        "domain": domain,
        "selection_split": "dev_only",
        "few_shot_pool": "train_10pct_only",
        "shots_per_class": int(shots_per_class),
        "seed": int(seed),
        "max_examples": max_examples,
        "data_hash": data["provenance"]["data_hash"],
        "ids_sha256": data["provenance"]["ids_sha256"],
        "git_commit": _git_commit(),
        "model_meta": model_meta or {},
        "pretraining_leakage_note": PRETRAINING_LEAKAGE_NOTE,
        "selected": {},
    }
    if not zs_df.empty:
        best_zs = zs_df.sort_values(["dev_selection_score", "invalid_output_rate"], ascending=[False, True]).iloc[0]
        selection["selected"]["zero_shot"] = {
            "strategy": str(best_zs["strategy"]),
            "prompt_style": str(best_zs["prompt_style"]),
            "dev_selection_score": float(best_zs["dev_selection_score"]),
            "invalid_output_rate": float(best_zs["invalid_output_rate"]),
        }
    if not fs_df.empty:
        best_fs = fs_df.sort_values(["dev_selection_score", "invalid_output_rate"], ascending=[False, True]).iloc[0]
        selection["selected"]["few_shot"] = {
            "strategy": str(best_fs["strategy"]),
            "prompt_style": str(best_fs["prompt_style"]),
            "dev_selection_score": float(best_fs["dev_selection_score"]),
            "invalid_output_rate": float(best_fs["invalid_output_rate"]),
        }
    sel_path = root / f"selection_llm_{domain}.json"
    atomic_json(sel_path, selection)
    return {"sweep_df": sweep_df, "selection": selection, "selection_path": sel_path}


def run_llm_test_eval(
    domain: str,
    selection_path: str | Path,
    predictor_fn: Callable[[Sequence[str]], Sequence[str]],
    *,
    tokenizer: Any = None,
    model_meta: Optional[Dict[str, Any]] = None,
    clean_dir: str | Path = "data/clean",
    subsets_dir: str | Path = "data/subsets_phase2",
    out_dir: str | Path = "results/phase2/llm",
    max_examples: Optional[int] = None,
    config_path: str | Path = "configs/task11_llm.yaml",
) -> Dict[str, Any]:
    """Run locked dev-selected zero-shot & few-shot configs once on held-out test."""
    cfg_bundle = load_llm_task_config(config_path)
    sel = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if sel.get("domain") != domain or sel.get("selection_split") != "dev_only":
        raise ValueError("Invalid LLM dev selection manifest")
    if sel.get("protocol_hash") != canonical_digest(cfg_bundle["protocol_cfg"]):
        raise ValueError("Selection protocol_hash mismatch")
    if max_examples is None and sel.get("max_examples") is not None:
        raise ValueError("Truncated dev sweep selection cannot be used for full test evaluation")

    data = load_llm_splits(
        domain,
        fraction=10,
        seed=int(sel["seed"]),
        clean_dir=clean_dir,
        subsets_dir=subsets_dir,
        include_test=True,
        max_examples=max_examples,
    )
    if sel["data_hash"] != data["provenance"]["data_hash"] or sel["ids_sha256"] != data["provenance"]["ids_sha256"]:
        raise ValueError("Selection data_hash/ids_sha256 mismatch")

    selector = FewShotSelector(
        data["train_texts"],
        data["train_labels"],
        data["labels"],
        seed=int(sel["seed"]),
        shots_per_class=int(sel["shots_per_class"]),
    )
    root = Path(out_dir) / domain
    root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    for mode_key, chosen in sel["selected"].items():
        strategy, style = chosen["strategy"], chosen["prompt_style"]
        res = evaluate_llm_split(
            "test",
            data["test_df"],
            data["test_texts"],
            data["test_labels"],
            domain,
            strategy,
            style,
            selector,
            predictor_fn,
            tokenizer=tokenizer,
        )
        run_dir = root / f"test_{mode_key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_csv(res["pred_frame"], run_dir / "predictions_test.csv")
        _atomic_csv(res["raw_frame"], run_dir / "raw_outputs_test.csv")
        atomic_json(
            run_dir / "metrics.json",
            {
                "domain": domain,
                "mode": mode_key,
                "strategy": strategy,
                "prompt_style": style,
                "n_eval": res["n_eval"],
                "invalid_output_count": res["invalid_output_count"],
                "invalid_output_rate": res["invalid_output_rate"],
                "test_metrics": res["view_metrics"],
                "probabilities_note": "Deterministic one-hot from greedy decoding; not calibrated posteriors.",
            },
        )
        atomic_json(
            run_dir / "metadata.json",
            {
                "protocol_version": cfg_bundle["protocol_cfg"]["protocol_version"],
                "domain": domain,
                "mode": mode_key,
                "strategy": strategy,
                "prompt_style": style,
                "shots": 0 if strategy == "zero_shot" else selector.total_shots,
                "selection_ref": str(Path(selection_path).resolve()),
                "data_provenance": data["provenance"],
                "model_meta": model_meta or sel.get("model_meta", {}),
                "git_commit": _git_commit(),
                "pretraining_leakage_risk": True,
                "pretraining_leakage_note": PRETRAINING_LEAKAGE_NOTE,
            },
        )
        for view_key, metrics in res["view_metrics"].items():
            view_name, map_method = view_key.split("/", 1)
            summary_rows.append(
                {
                    "domain": domain,
                    "mode": mode_key,
                    "strategy": strategy,
                    "prompt_style": style,
                    "label_view": view_name,
                    "map_method": map_method,
                    "invalid_output_rate": res["invalid_output_rate"],
                    "elapsed_seconds": res["elapsed_seconds"],
                    "throughput_samples_per_sec": res["throughput_samples_per_sec"],
                    **metrics,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    _atomic_csv(summary_df, root / f"test_summary_{domain}.csv")
    return {"summary_df": summary_df, "out_dir": root}


def benchmark_llm_inference(
    classifier: HuggingFaceLLMClassifier,
    texts: Sequence[str],
    domain: str,
    *,
    n_b1: int = 50,
    large_batch_size: int = 16,
    n_large: int = 160,
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Measure peak VRAM, batch=1 latency, large-batch throughput, and GPU-h/1k samples."""
    import torch

    prompts = [
        render_prompt_text(classifier.tokenizer, build_llm_messages(t, domain, [], prompt_style="concise"))
        for t in list(texts)[: max(n_b1, n_large)]
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    classifier.generate_batch(prompts[:1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Batch = 1 latency
    prev_bs = classifier.batch_size
    classifier.batch_size = 1
    b1_times: List[float] = []
    for p in prompts[: min(n_b1, len(prompts))]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        classifier.generate_batch([p])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        b1_times.append(time.perf_counter() - t0)

    # Large batch throughput
    classifier.batch_size = int(large_batch_size)
    batch_prompts = prompts[: min(n_large, len(prompts))]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    classifier.generate_batch(batch_prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    large_elapsed = time.perf_counter() - t0
    classifier.batch_size = prev_bs

    n_proc = len(batch_prompts)
    throughput = n_proc / large_elapsed if large_elapsed > 0 else 0.0
    gpu_hours_1k = (large_elapsed / n_proc) * 1000.0 / 3600.0 if n_proc > 0 else 0.0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    cost = {
        **classifier.metadata(),
        "domain": domain,
        "peak_vram_mb": float(peak_vram_mb),
        "n_b1": len(b1_times),
        "latency_batch1_mean_ms": float(np.mean(b1_times) * 1000.0) if b1_times else 0.0,
        "latency_batch1_p50_ms": float(np.median(b1_times) * 1000.0) if b1_times else 0.0,
        "latency_batch1_p95_ms": float(np.percentile(b1_times, 95) * 1000.0) if b1_times else 0.0,
        "large_batch_size": int(large_batch_size),
        "n_large": n_proc,
        "large_batch_seconds": float(large_elapsed),
        "throughput_samples_per_sec": float(throughput),
        "gpu_hours_per_1000_samples": float(gpu_hours_1k),
    }
    if out_path is not None:
        atomic_json(out_path, cost)
    return cost


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["dev-sweep", "eval-test", "benchmark"])
    parser.add_argument("--config", default="configs/task11_llm.yaml")
    parser.add_argument("--domain", choices=["vsmec", "neuesc"], required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32", "4bit", "8bit"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--selection-path")
    parser.add_argument("--out-dir", default="results/phase2/llm")
    parser.add_argument("--no-verify-license", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_llm_task_config(args.config)["llm"]
    model_name = args.model_name or cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    precision = args.precision or cfg.get("precision", "bf16")
    batch_size = args.batch_size or int(cfg.get("batch_size", 16))
    clf = HuggingFaceLLMClassifier(
        model_name=model_name,
        precision=precision,
        max_input_length=int(cfg.get("max_input_length", 1024)),
        max_new_tokens=int(cfg.get("max_new_tokens", 16)),
        batch_size=batch_size,
        verify_license=not args.no_verify_license,
    )

    if args.action == "dev-sweep":
        shots = int((cfg.get("shots_per_class") or {}).get(args.domain, 1))
        run_llm_dev_sweep(
            args.domain,
            clf.generate_batch,
            tokenizer=clf.tokenizer,
            model_meta=clf.metadata(),
            strategies=tuple(cfg.get("strategies", STRATEGIES)),
            prompt_styles=tuple(cfg.get("prompt_styles", PROMPT_STYLES)),
            shots_per_class=shots,
            seed=args.seed,
            out_dir=args.out_dir,
            max_examples=args.max_examples,
            config_path=args.config,
        )
    elif args.action == "eval-test":
        sel_path = args.selection_path or str(Path(args.out_dir) / args.domain / f"selection_llm_{args.domain}.json")
        run_llm_test_eval(
            args.domain,
            sel_path,
            clf.generate_batch,
            tokenizer=clf.tokenizer,
            model_meta=clf.metadata(),
            out_dir=args.out_dir,
            max_examples=args.max_examples,
            config_path=args.config,
        )
    elif args.action == "benchmark":
        data = load_llm_splits(args.domain, include_test=False, seed=args.seed)
        out = Path(args.out_dir) / f"cost_llm_{args.domain}.json"
        cost = benchmark_llm_inference(clf, data["dev_texts"], args.domain, large_batch_size=batch_size, out_path=out)
        print(json.dumps(cost, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
