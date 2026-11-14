"""The single configuration contract used by training, selection and analysis."""
import hashlib
import json
from pathlib import Path

import yaml

DIRECTIONS = {"SA": ("vsmec", "neuesc", 3), "SB": ("neu3", "vsmec", 7),
              "C0N": ("neu3", "neuesc", 3), "C0V": ("vsmec", "vsmec", 7)}
PRIMARY = [("P1", "PhoBERT", "SA", "V3-NEU", "macro_f1"),
           ("P2", "ViSoBERT", "SA", "V3-NEU", "macro_f1"),
           ("P3", "PhoBERT", "SB", "V7-VSMEC", "macro_f1"),
           ("P4", "ViSoBERT", "SB", "V7-VSMEC", "macro_f1")]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path="configs/seq.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected_training = dict(batch_size=16, gradient_accumulation_steps=2, epochs=10, patience=5,
                             warmup_ratio=0.06, weight_decay=0.01, max_grad_norm=1.0, dropout=0.1,
                             max_length=256, precision="bf16", strict_precision=True, evals_per_epoch=2)
    if cfg.get("protocol_version") != "seq-v1" or cfg["training"] != expected_training:
        raise ValueError("Training profile differs from frozen seq-v1; use a separately reviewed protocol version")
    if set(cfg["backbones"]) != {"PhoBERT", "ViSoBERT"}:
        raise ValueError("Core protocol requires both canonical backbone aliases")
    if not cfg["c0_seeds"] or len(set(cfg["c0_seeds"])) != len(cfg["c0_seeds"]) or not set(cfg["c0_seeds"]).issubset(range(5)):
        raise ValueError("C0 seed IDs must be a nonempty unique subset of core seeds")
    if cfg["c0_lr_mode"] != "reuse":
        raise ValueError("seq-v1 fixes c0_lr_mode=reuse; change protocol version for a separate C0 sweep")
    if cfg["seeds"] != list(range(5)) or cfg["sweep_seed"] != 0:
        raise ValueError("Core protocol requires seeds0..4 and sweep_seed0")
    if sorted(float(x) for x in cfg["lr_grid"]) != [1e-5, 2e-5, 3e-5]:
        raise ValueError("seq-v1 requires the complete fixed LR grid")
    actual = [tuple(item[k] for k in ("id", "backbone", "direction", "label_view", "stat")) for item in cfg["primary"]]
    if actual != PRIMARY:
        raise ValueError("Primary family differs from frozen seq-v1 comparisons")
    return cfg


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
