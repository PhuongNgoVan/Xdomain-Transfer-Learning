"""Task 11 LLM zero-shot and few-shot evaluation on Phase 2 10% splits."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

_NEGATION_PREFIX_RE = re.compile(
    r"(?:không\s+phải(?:\s+là)?|chứ\s+không(?:\s+phải)?|không\s+thể\s+là|không|not|isn't|is\s+not|never)"
    r"(?:\s+(?:nhãn|label))?[\s\"'`*:\-\[\]()]*$",
    flags=re.IGNORECASE,
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
        # Shuffle fixed examples deterministically so the final class never sits last by default.
        perm = rng.permutation(len(fixed)).tolist()
        self._fixed_indices = [fixed[i] for i in perm]

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
                order = np.argsort(row, kind="mergesort")[-self.total_shots :]
                chosen = order.tolist()
            elif strategy == "balanced_tfidf_knn":
                picked: List[Tuple[float, int]] = []
                for cls_id in range(self.num_classes):
                    pool = np.asarray(self._by_class[cls_id], dtype=np.int64)
                    cls_scores = row[pool]
                    take = min(self.shots_per_class, len(pool))
                    top_local = np.argsort(cls_scores, kind="mergesort")[-take:]
                    for local_idx in top_local:
                        idx = int(pool[local_idx])
                        picked.append((float(row[idx]), idx))
                picked.sort(key=lambda item: (item[0], item[1]))
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


def _count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    if tokenizer is None:
        return len(prompt.split())
    if callable(tokenizer):
        try:
            enc = tokenizer(prompt, add_special_tokens=False, truncation=False)
            if isinstance(enc, Mapping) and "input_ids" in enc:
                return len(enc["input_ids"])
        except TypeError:
            pass
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(prompt, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(prompt))
    return len(prompt.split())


def prepare_safe_llm_prompt(
    tokenizer: Any,
    query_text: str,
    domain: str,
    examples: Sequence[Tuple[str, str]],
    prompt_style: str = "concise",
    *,
    max_input_length: int = 2048,
    max_example_chars: int = 600,
) -> Tuple[str, int, bool]:
    """Render prompt and trim only few-shot examples if needed so the target query is never truncated."""
    limit = int(max_input_length)
    if limit <= 0:
        raise ValueError("max_input_length must be positive")

    base_prompt = render_prompt_text(
        tokenizer, build_llm_messages(query_text, domain, [], prompt_style=prompt_style)
    )
    base_tokens = _count_prompt_tokens(tokenizer, base_prompt)
    if base_tokens > limit:
        raise ValueError(
            f"Target query prompt ({base_tokens} tokens) exceeds max_input_length={limit}; "
            "refusing to right-truncate the query."
        )

    trimmed = False
    cur_examples: List[Tuple[str, str]] = []
    for ex_text, ex_label in examples:
        txt = str(ex_text)
        if max_example_chars and len(txt) > int(max_example_chars):
            txt = txt[: max(1, int(max_example_chars) - 3)] + "..."
            trimmed = True
        cur_examples.append((txt, str(ex_label)))

    rendered = render_prompt_text(
        tokenizer, build_llm_messages(query_text, domain, cur_examples, prompt_style=prompt_style)
    )
    tok_count = _count_prompt_tokens(tokenizer, rendered)
    if tok_count <= limit or not cur_examples:
        return rendered, tok_count, trimmed

    # Proportionally shorten few-shot example texts while preserving target query & labels verbatim.
    for cap in (240, 140, 80, 40, 20):
        trimmed = True
        shrunk = [
            ((txt[:cap] + "...") if len(txt) > cap else txt, lbl)
            for txt, lbl in cur_examples
        ]
        rendered = render_prompt_text(
            tokenizer, build_llm_messages(query_text, domain, shrunk, prompt_style=prompt_style)
        )
        tok_count = _count_prompt_tokens(tokenizer, rendered)
        if tok_count <= limit:
            return rendered, tok_count, True

    # Drop oldest/lowest-ranked examples from the front if text caps alone still exceed limit.
    while len(cur_examples) > 1:
        cur_examples = cur_examples[1:]
        trimmed = True
        shrunk = [((txt[:80] + "...") if len(txt) > 80 else txt, lbl) for txt, lbl in cur_examples]
        rendered = render_prompt_text(
            tokenizer, build_llm_messages(query_text, domain, shrunk, prompt_style=prompt_style)
        )
        tok_count = _count_prompt_tokens(tokenizer, rendered)
        if tok_count <= limit:
            return rendered, tok_count, True

    raise ValueError(
        f"Prompt ({tok_count} tokens) still exceeds max_input_length={limit} after trimming examples."
    )


def parse_llm_label(raw_output: str, domain: str) -> Tuple[int, str, bool]:
    """Strictly parse generated text into target label ID, canonical name, and validity flag."""
    labels = DOMAIN_LABELS[domain]
    lower_map = {name.lower(): (idx, name) for idx, name in enumerate(labels)}
    fb = FALLBACK_LABEL_ID[domain]
    text = str(raw_output or "").strip()
    if not text:
        return fb, labels[fb], False

    all_hits: List[Tuple[int, int, str]] = []
    for idx, name in enumerate(labels):
        for m in re.finditer(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            all_hits.append((m.start(), idx, name))

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""
    cleaned = re.sub(r'^[\s"\'`*:\-\[\]().,;]+|[\s"\'`*:\-\[\]().,;]+$', "", first_line).lower()
    if cleaned in lower_map:
        idx, name = lower_map[cleaned]
        other_ids = {h_idx for _, h_idx, _ in all_hits}
        if not other_ids or other_ids == {idx}:
            return idx, name, True
        return fb, labels[fb], False

    # JSON {"label": "..."} or {"nhãn": "..."}
    json_matches = re.findall(r'"?(?:label|nhãn)"?\s*:\s*"?([a-zA-Z]+)"?', text, flags=re.IGNORECASE)
    json_ids = {lower_map[cand.lower()] for cand in json_matches if cand.lower() in lower_map}
    if len(json_ids) == 1:
        idx, name = next(iter(json_ids))
        return idx, name, True
    if len(json_ids) > 1:
        return fb, labels[fb], False

    # Explicit line-anchored declaration ("Nhãn: positive", "Label: Anger")
    decl_matches = re.findall(
        r'(?im)^\s*(?:nhãn|label|đáp án|answer)\s*[:=]\s*["\'`*]*([a-zA-Z]+)["\'`*.\s]*$',
        text,
    )
    decl_ids = {lower_map[cand.lower()] for cand in decl_matches if cand.lower() in lower_map}
    if len(decl_ids) == 1:
        idx, name = next(iter(decl_ids))
        return idx, name, True
    if len(decl_ids) > 1:
        return fb, labels[fb], False

    # Fallback boundary scan: valid ONLY when a single unambiguous un-negated label appears.
    unique_ids = {idx for _, idx, _ in all_hits}
    if len(unique_ids) != 1:
        return fb, labels[fb], False

    all_hits.sort(key=lambda item: item[0])
    pos, idx, name = all_hits[0]
    prefix = text[max(0, pos - 32) : pos]
    if _NEGATION_PREFIX_RE.search(prefix):
        return fb, labels[fb], False
    return idx, name, True


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


def _count_causal_lm_params(model: Any) -> Tuple[int, Optional[float]]:
    """Use `model.num_parameters()` so packed 4-bit/8-bit weights report true parameter counts."""
    if hasattr(model, "num_parameters"):
        total_params = int(model.num_parameters())
    else:
        total_params = int(sum(p.numel() for p in model.parameters()))
    footprint_mb: Optional[float] = None
    if hasattr(model, "get_memory_footprint"):
        try:
            footprint_mb = float(model.get_memory_footprint() / (1024 ** 2))
        except Exception:
            footprint_mb = None
    return total_params, footprint_mb


class HuggingFaceLLMClassifier:
    """Greedy-decoding zero/few-shot classifier for open-weight CausalLMs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        precision: str = "bf16",
        max_input_length: int = 2048,
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
        self.total_params, self.memory_footprint_mb = _count_causal_lm_params(self.model)
        self.trainable_params = 0

    def generate_batch(self, prompts: Sequence[str], *, quiet: bool = False) -> List[str]:
        import torch

        device = next(self.model.parameters()).device
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        restored = [""] * len(prompts)
        total_batches = max(1, math.ceil(len(order) / self.batch_size))
        with torch.inference_mode():
            for batch_idx, start in enumerate(range(0, len(order), self.batch_size), 1):
                idxs = order[start : start + self.batch_size]
                batch_prompts = [prompts[i] for i in idxs]
                raw_lens = [len(ids) for ids in self.tokenizer(batch_prompts, padding=False, truncation=False)["input_ids"]]
                if max(raw_lens, default=0) > self.max_input_length:
                    raise ValueError(
                        f"Prompt token length {max(raw_lens)} exceeds max_input_length={self.max_input_length}; "
                        "refusing to right-truncate target query."
                    )
                enc = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
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
                if not quiet and (batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches):
                    print(f"[llm] batch {batch_idx}/{total_batches} ({min(start + self.batch_size, len(order))}/{len(order)})", flush=True)
        return restored

    def metadata(self) -> Dict[str, Any]:
        import torch

        meta = {
            **self.hub_meta,
            "precision": self.precision,
            "quantization": self.precision if self.precision in ("4bit", "8bit") else "none",
            "temperature": 0.0,
            "do_sample": False,
            "max_input_length": int(self.max_input_length),
            "total_params": int(self.total_params),
            "trainable_params": int(self.trainable_params),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        if self.memory_footprint_mb is not None:
            meta["memory_footprint_mb"] = float(self.memory_footprint_mb)
        return meta


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
    *,
    max_input_length: int = 2048,
    max_example_chars: int = 600,
) -> Dict[str, Any]:
    """Run deterministic LLM prediction and score all domain views on one split."""
    examples_per_row = selector.select_batch(texts, strategy)
    prompts: List[str] = []
    prompt_tokens: List[int] = []
    trimmed_count = 0
    for text, exs in zip(texts, examples_per_row):
        rendered, n_tok, was_trimmed = prepare_safe_llm_prompt(
            tokenizer,
            text,
            domain,
            exs,
            prompt_style=prompt_style,
            max_input_length=max_input_length,
            max_example_chars=max_example_chars,
        )
        prompts.append(rendered)
        prompt_tokens.append(n_tok)
        trimmed_count += int(was_trimmed)

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
        else [
            ("V7-VSMEC", "none"),
            ("V5-VSMEC", "none"),
            ("V3-VSMEC", "sum"),
            ("V3-VSMEC", "argmax"),
            ("V3-VSMEC-strict", "sum"),
            ("V3-VSMEC-strict", "argmax"),
        ]
    )
    view_metrics = {f"{v}/{m}": score_view(pred_frame, v, m) for v, m in views}
    raw_df = pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "y_true": list(labels),
            "y_pred": parsed_ids,
            "pred_label": parsed_names,
            "valid_parse": valid_flags,
            "prompt_tokens": prompt_tokens,
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
        "max_prompt_tokens": int(max(prompt_tokens, default=0)),
        "mean_prompt_tokens": float(np.mean(prompt_tokens)) if prompt_tokens else 0.0,
        "trimmed_example_prompts": int(trimmed_count),
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
    """Evaluate zero-shot and few-shot strategies on dev ONLY, supporting artifact SKIP, and lock selection JSON."""
    cfg_bundle = load_llm_task_config(config_path)
    llm_cfg = cfg_bundle["llm"]
    max_input_length = int(llm_cfg.get("max_input_length", 2048))
    max_example_chars = int(llm_cfg.get("max_example_chars", 600))

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
    proto_hash = canonical_digest(cfg_bundle["protocol_cfg"])
    code_sha = sha256_file(Path(__file__))
    cfg_sha = sha256_file(Path(config_path))

    rows: List[Dict[str, Any]] = []
    skipped_tags: List[str] = []
    for style in prompt_styles:
        for strategy in strategies:
            tag = f"{strategy}__{style}"
            pred_path = root / f"dev_pred_{tag}.csv"
            raw_path = root / f"dev_raw_{tag}.csv"
            comp_path = root / f"dev_completion_{tag}.json"
            tag_fp = canonical_digest(
                {
                    "protocol_hash": proto_hash,
                    "domain": domain,
                    "strategy": strategy,
                    "prompt_style": style,
                    "shots_per_class": int(shots_per_class),
                    "seed": int(seed),
                    "max_examples": max_examples,
                    "max_input_length": max_input_length,
                    "max_example_chars": max_example_chars,
                    "data_hash": data["provenance"]["data_hash"],
                    "ids_sha256": data["provenance"]["ids_sha256"],
                    "model_meta": {
                        "model_name": (model_meta or {}).get("model_name"),
                        "precision": (model_meta or {}).get("precision"),
                    },
                    "code_sha256": code_sha,
                    "config_sha256": cfg_sha,
                }
            )
            if comp_path.is_file():
                comp = json.loads(comp_path.read_text(encoding="utf-8"))
                if comp.get("fingerprint") != tag_fp:
                    raise ValueError(f"Dev sweep fingerprint mismatch for {domain}/{tag}")
                artifacts = comp.get("artifacts", {})
                if (
                    comp.get("complete") is True
                    and pred_path.is_file()
                    and raw_path.is_file()
                    and artifacts.get(pred_path.name) == sha256_file(pred_path)
                    and artifacts.get(raw_path.name) == sha256_file(raw_path)
                    and isinstance(comp.get("row"), dict)
                ):
                    rows.append(comp["row"])
                    skipped_tags.append(tag)
                    continue

            atomic_json(comp_path, {"complete": False, "fingerprint": tag_fp})
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
                max_input_length=max_input_length,
                max_example_chars=max_example_chars,
            )
            _atomic_csv(res["pred_frame"], pred_path)
            _atomic_csv(res["raw_frame"], raw_path)
            prim = res["view_metrics"][primary_view]
            row = {
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
                "max_prompt_tokens": res["max_prompt_tokens"],
                "mean_prompt_tokens": res["mean_prompt_tokens"],
                "trimmed_example_prompts": res["trimmed_example_prompts"],
                "elapsed_seconds": res["elapsed_seconds"],
                "data_hash": data["provenance"]["data_hash"],
                "ids_sha256": data["provenance"]["ids_sha256"],
                "max_examples": max_examples,
            }
            atomic_json(
                comp_path,
                {
                    "complete": True,
                    "fingerprint": tag_fp,
                    "artifacts": {
                        pred_path.name: sha256_file(pred_path),
                        raw_path.name: sha256_file(raw_path),
                    },
                    "row": row,
                },
            )
            rows.append(row)

    sweep_df = pd.DataFrame(rows)
    _atomic_csv(sweep_df, root / f"dev_sweep_{domain}.csv")

    style_rank = {s: i for i, s in enumerate(PROMPT_STYLES)}
    strat_rank = {s: i for i, s in enumerate(STRATEGIES)}

    def _pick_best(sub: pd.DataFrame) -> pd.Series:
        ordered = sub.assign(
            _s_rank=sub["prompt_style"].map(lambda x: style_rank.get(str(x), 99)),
            _t_rank=sub["strategy"].map(lambda x: strat_rank.get(str(x), 99)),
        ).sort_values(
            ["dev_selection_score", "invalid_output_rate", "_s_rank", "_t_rank"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        return ordered.iloc[0]

    zs_df = sweep_df.loc[sweep_df["strategy"].eq("zero_shot")]
    fs_df = sweep_df.loc[sweep_df["strategy"].ne("zero_shot")]
    selection: Dict[str, Any] = {
        "protocol_version": cfg_bundle["protocol_cfg"]["protocol_version"],
        "protocol_hash": proto_hash,
        "domain": domain,
        "selection_split": "dev_only",
        "few_shot_pool": "train_10pct_only",
        "shots_per_class": int(shots_per_class),
        "seed": int(seed),
        "max_examples": max_examples,
        "max_input_length": max_input_length,
        "data_hash": data["provenance"]["data_hash"],
        "ids_sha256": data["provenance"]["ids_sha256"],
        "git_commit": _git_commit(),
        "model_meta": model_meta or {},
        "pretraining_leakage_note": PRETRAINING_LEAKAGE_NOTE,
        "selected": {},
    }
    if not zs_df.empty:
        best_zs = _pick_best(zs_df)
        selection["selected"]["zero_shot"] = {
            "strategy": str(best_zs["strategy"]),
            "prompt_style": str(best_zs["prompt_style"]),
            "dev_selection_score": float(best_zs["dev_selection_score"]),
            "invalid_output_rate": float(best_zs["invalid_output_rate"]),
        }
    if not fs_df.empty:
        best_fs = _pick_best(fs_df)
        selection["selected"]["few_shot"] = {
            "strategy": str(best_fs["strategy"]),
            "prompt_style": str(best_fs["prompt_style"]),
            "dev_selection_score": float(best_fs["dev_selection_score"]),
            "invalid_output_rate": float(best_fs["invalid_output_rate"]),
        }
    sel_path = root / f"selection_llm_{domain}.json"
    atomic_json(sel_path, selection)
    return {
        "sweep_df": sweep_df,
        "selection": selection,
        "selection_path": sel_path,
        "skipped_tags": skipped_tags,
    }


def _verify_llm_model_meta_compatibility(
    sel_meta: Mapping[str, Any],
    cur_meta: Optional[Mapping[str, Any]],
) -> None:
    if not cur_meta or not sel_meta:
        return
    for field in ("model_name", "precision"):
        sel_val = sel_meta.get(field)
        cur_val = cur_meta.get(field)
        if sel_val is not None and cur_val is not None and str(sel_val) != str(cur_val):
            raise ValueError(
                f"Selection {field} mismatch: selection has {sel_val!r}, runtime has {cur_val!r}"
            )


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
    """Run locked dev-selected zero-shot & few-shot configs once on held-out test with completion/SKIP."""
    cfg_bundle = load_llm_task_config(config_path)
    llm_cfg = cfg_bundle["llm"]
    max_input_length = int(llm_cfg.get("max_input_length", 2048))
    max_example_chars = int(llm_cfg.get("max_example_chars", 600))

    sel = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if sel.get("domain") != domain or sel.get("selection_split") != "dev_only":
        raise ValueError("Invalid LLM dev selection manifest")
    if sel.get("protocol_hash") != canonical_digest(cfg_bundle["protocol_cfg"]):
        raise ValueError("Selection protocol_hash mismatch")
    if max_examples is None and sel.get("max_examples") is not None:
        raise ValueError("Truncated dev sweep selection cannot be used for full test evaluation")
    _verify_llm_model_meta_compatibility(sel.get("model_meta") or {}, model_meta)

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
    skipped_modes: List[str] = []
    sel_sha = sha256_file(Path(selection_path))
    code_sha = sha256_file(Path(__file__))
    effective_meta = model_meta or sel.get("model_meta", {})

    for mode_key, chosen in sel["selected"].items():
        strategy, style = chosen["strategy"], chosen["prompt_style"]
        run_dir = root / f"test_{mode_key}"
        comp_path = run_dir / "completion.json"
        mode_fp = canonical_digest(
            {
                "protocol_hash": sel["protocol_hash"],
                "domain": domain,
                "mode": mode_key,
                "strategy": strategy,
                "prompt_style": style,
                "selection_sha256": sel_sha,
                "data_hash": data["provenance"]["data_hash"],
                "ids_sha256": data["provenance"]["ids_sha256"],
                "max_examples": max_examples,
                "max_input_length": max_input_length,
                "max_example_chars": max_example_chars,
                "model_meta": {
                    "model_name": (effective_meta or {}).get("model_name"),
                    "precision": (effective_meta or {}).get("precision"),
                },
                "code_sha256": code_sha,
            }
        )
        req_files = ("predictions_test.csv", "raw_outputs_test.csv", "metrics.json", "metadata.json")
        if comp_path.is_file():
            comp = json.loads(comp_path.read_text(encoding="utf-8"))
            if comp.get("fingerprint") != mode_fp:
                raise ValueError(f"LLM test completion fingerprint mismatch for {domain}/{mode_key}")
            artifacts = comp.get("artifacts", {})
            if (
                comp.get("complete") is True
                and set(req_files).issubset(artifacts)
                and all((run_dir / fname).is_file() and sha256_file(run_dir / fname) == artifacts[fname] for fname in req_files)
                and isinstance(comp.get("summary_rows"), list)
            ):
                summary_rows.extend(comp["summary_rows"])
                skipped_modes.append(mode_key)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(comp_path, {"complete": False, "fingerprint": mode_fp})
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
            max_input_length=max_input_length,
            max_example_chars=max_example_chars,
        )
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
                "max_prompt_tokens": res["max_prompt_tokens"],
                "mean_prompt_tokens": res["mean_prompt_tokens"],
                "trimmed_example_prompts": res["trimmed_example_prompts"],
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
                "selection_sha256": sel_sha,
                "data_provenance": data["provenance"],
                "model_meta": effective_meta,
                "git_commit": _git_commit(),
                "fingerprint": mode_fp,
                "pretraining_leakage_risk": True,
                "pretraining_leakage_note": PRETRAINING_LEAKAGE_NOTE,
            },
        )
        mode_rows: List[Dict[str, Any]] = []
        for view_key, metrics in res["view_metrics"].items():
            view_name, map_method = view_key.split("/", 1)
            mode_rows.append(
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
        artifacts = {fname: sha256_file(run_dir / fname) for fname in req_files}
        atomic_json(
            comp_path,
            {
                "complete": True,
                "fingerprint": mode_fp,
                "artifacts": artifacts,
                "summary_rows": mode_rows,
            },
        )
        summary_rows.extend(mode_rows)

    summary_df = pd.DataFrame(summary_rows)
    _atomic_csv(summary_df, root / f"test_summary_{domain}.csv")
    return {"summary_df": summary_df, "out_dir": root, "skipped_modes": skipped_modes}


def _invoke_generate_quiet(classifier: Any, prompts: Sequence[str]) -> Sequence[str]:
    sig = inspect.signature(classifier.generate_batch)
    if "quiet" in sig.parameters:
        return classifier.generate_batch(prompts, quiet=True)
    return classifier.generate_batch(prompts)


def _benchmark_prompt_set(
    classifier: Any,
    prompts: Sequence[str],
    *,
    n_b1: int,
    large_batch_size: int,
    n_large: int,
) -> Dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    _invoke_generate_quiet(classifier, prompts[:1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prev_bs = classifier.batch_size
    classifier.batch_size = 1
    b1_times: List[float] = []
    for p in prompts[: min(int(n_b1), len(prompts))]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _invoke_generate_quiet(classifier, [p])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        b1_times.append(time.perf_counter() - t0)

    classifier.batch_size = int(large_batch_size)
    batch_prompts = prompts[: min(int(n_large), len(prompts))]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _invoke_generate_quiet(classifier, batch_prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    large_elapsed = time.perf_counter() - t0
    classifier.batch_size = prev_bs

    n_proc = len(batch_prompts)
    throughput = n_proc / large_elapsed if large_elapsed > 0 else 0.0
    gpu_hours_1k = (large_elapsed / n_proc) * 1000.0 / 3600.0 if (n_proc > 0 and torch.cuda.is_available()) else 0.0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return {
        "peak_vram_mb": float(peak_vram_mb),
        "n_b1": len(b1_times),
        "latency_batch1_mean_ms": float(np.mean(b1_times) * 1000.0) if b1_times else 0.0,
        "latency_batch1_p50_ms": float(np.median(b1_times) * 1000.0) if b1_times else 0.0,
        "latency_batch1_p95_ms": float(np.percentile(b1_times, 95) * 1000.0) if b1_times else 0.0,
        "large_batch_size": int(large_batch_size),
        "n_large": int(n_proc),
        "large_batch_seconds": float(large_elapsed),
        "throughput_samples_per_sec": float(throughput),
        "gpu_hours_per_1000_samples": float(gpu_hours_1k),
    }


def benchmark_llm_inference(
    classifier: Any,
    texts: Sequence[str],
    domain: str,
    *,
    n_b1: int = 50,
    large_batch_size: int = 16,
    n_large: int = 160,
    out_path: Optional[str | Path] = None,
    selection_path: Optional[str | Path] = None,
    clean_dir: str | Path = "data/clean",
    subsets_dir: str | Path = "data/subsets_phase2",
    max_input_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure peak VRAM, batch=1 latency, large-batch throughput, and GPU-h/1k for zero-shot & locked few-shot."""
    sample_texts = list(texts)[: max(int(n_b1), int(n_large))]
    limit = int(max_input_length or getattr(classifier, "max_input_length", 2048))

    zs_prompts = [
        prepare_safe_llm_prompt(
            classifier.tokenizer, t, domain, [], prompt_style="concise", max_input_length=limit
        )[0]
        for t in sample_texts
    ]
    modes: Dict[str, Dict[str, Any]] = {}
    if selection_path is not None and Path(selection_path).is_file():
        sel = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        sel_limit = int(sel.get("max_input_length", limit))
        splits = load_llm_splits(
            domain,
            fraction=10,
            seed=int(sel["seed"]),
            clean_dir=clean_dir,
            subsets_dir=subsets_dir,
            include_test=False,
        )
        selector = FewShotSelector(
            splits["train_texts"],
            splits["train_labels"],
            splits["labels"],
            seed=int(sel["seed"]),
            shots_per_class=int(sel["shots_per_class"]),
        )
        for mode_key, chosen in (sel.get("selected") or {}).items():
            strategy = str(chosen["strategy"])
            style = str(chosen["prompt_style"])
            ex_batches = selector.select_batch(sample_texts, strategy)
            mode_prompts = [
                prepare_safe_llm_prompt(
                    classifier.tokenizer, t, domain, exs, prompt_style=style, max_input_length=sel_limit
                )[0]
                for t, exs in zip(sample_texts, ex_batches)
            ]
            modes[mode_key] = {
                "strategy": strategy,
                "prompt_style": style,
                **_benchmark_prompt_set(
                    classifier, mode_prompts, n_b1=n_b1, large_batch_size=large_batch_size, n_large=n_large
                ),
            }
    if "zero_shot" not in modes:
        modes["zero_shot"] = {
            "strategy": "zero_shot",
            "prompt_style": "concise",
            **_benchmark_prompt_set(
                classifier, zs_prompts, n_b1=n_b1, large_batch_size=large_batch_size, n_large=n_large
            ),
        }

    primary = modes["zero_shot"]
    cost: Dict[str, Any] = {
        **classifier.metadata(),
        "domain": domain,
        **{k: v for k, v in primary.items() if k not in ("strategy", "prompt_style")},
        "modes": modes,
    }
    if "few_shot" in modes:
        fs = modes["few_shot"]
        cost.update(
            {
                "few_shot_strategy": fs["strategy"],
                "few_shot_prompt_style": fs["prompt_style"],
                "few_shot_peak_vram_mb": fs["peak_vram_mb"],
                "few_shot_latency_batch1_mean_ms": fs["latency_batch1_mean_ms"],
                "few_shot_throughput_samples_per_sec": fs["throughput_samples_per_sec"],
                "few_shot_gpu_hours_per_1000_samples": fs["gpu_hours_per_1000_samples"],
            }
        )
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
        max_input_length=int(cfg.get("max_input_length", 2048)),
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
        default_sel = Path(args.out_dir) / args.domain / f"selection_llm_{args.domain}.json"
        sel_path = args.selection_path or (str(default_sel) if default_sel.is_file() else None)
        cost = benchmark_llm_inference(
            clf,
            data["dev_texts"],
            args.domain,
            large_batch_size=batch_size,
            out_path=out,
            selection_path=sel_path,
        )
        print(json.dumps(cost, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
