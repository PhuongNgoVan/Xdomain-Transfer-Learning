"""Prepare raw text inputs; keep held-out test outside sweep/smoke execution."""
from src.data.preprocess import preprocess_text
from src.phase2.audit_task01_02 import check_underscore_leakage
from src.phase2.protocol import canonical_digest
from src.phase2.subsets import load_phase2_target_splits


def load_phase2_prepared_data(direction, fraction, seed, backbone, protocol_cfg, *,
                              run_mode="main", clean_dir="data/clean",
                              subsets_dir="data/subsets_phase2", max_examples=None):
    if run_mode not in ("main", "sweep", "smoke"):
        raise ValueError("Invalid run_mode")
    if backbone not in ("ViSoBERT", "XLMR"):
        raise ValueError("Phase 2 supports ViSoBERT/XLMR raw text only")
    spec = protocol_cfg["backbones"][backbone]
    if spec["text_col"] != "text" or spec["emoji_mode"] != "keep":
        raise ValueError("Expected text_col='text', emoji_mode='keep'")
    if max_examples is not None and int(max_examples) <= 0:
        raise ValueError("max_examples must be positive")
    target = protocol_cfg["directions"][direction]["target"]
    train, dev, test, meta = load_phase2_target_splits(
        target, fraction, seed, clean_dir=clean_dir, subsets_dir=subsets_dir,
        include_test=run_mode == "main")
    result = {"target_domain": target, "subset_meta": meta}
    hashes = {}
    for split, frame in (("train", train), ("dev", dev), ("test", test)):
        texts = labels = None
        if frame is not None:
            if max_examples is not None:
                frame = frame.iloc[:int(max_examples)].copy()
            frame = frame.reset_index(drop=True)
            raw = frame["text"].astype(str).tolist()
            texts = [preprocess_text(t, emoji_mode="keep") for t in raw]
            if not texts or any(not t.strip() for t in texts):
                raise ValueError(f"Empty {split} input text")
            if check_underscore_leakage(raw, texts)["status"] != "PASS":
                raise ValueError(f"Underscore leakage in {split}")
            labels = frame["label"].astype(int).tolist()
            hashes[split] = canonical_digest(texts)
        result.update({f"{split}_df": frame, f"{split}_texts": texts, f"{split}_labels": labels})
    result["provenance"] = {**meta, "input_texts_sha256": hashes,
                            "max_examples": max_examples, "test_integrity_checked": run_mode == "main"}
    return result
