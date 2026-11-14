"""Test-only data loading and fixed label views; no tokenization of train/dev."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.zs.mapping import map_labels7

VIEW_LABELS = {"V3-NEU": (0, 1, 2), "V3-VSMEC": (0, 1, 2), "V3-VSMEC-strict": (0, 2), "V7-VSMEC": tuple(range(7))}


def strict_mask(labels7):
    array = np.asarray(labels7)
    if not np.isin(array, np.arange(7)).all():
        raise ValueError("Strict membership requires original emotion labels in 0..6")
    return array <= 4


def load_test_frame(data_dir, domain, max_examples=None):
    if domain not in ("vsmec", "neuesc"):
        raise ValueError(f"Unknown evaluation domain: {domain}")
    path = Path(data_dir) / ("vsmec_test.csv" if domain == "vsmec" else "neuesc_100.csv")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"id", "text", "text_seg", "label", "subset"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing evaluation columns: {sorted(required - set(frame.columns))}")
    # Filtering precedes validation/normalization/tokenization of evaluation texts.
    frame = frame.loc[frame["subset"].eq("test")].copy().reset_index(drop=True)
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError("max_examples must be positive")
        frame = frame.iloc[:int(max_examples)].copy()
    if frame.empty or frame["id"].duplicated().any() or frame["id"].isna().any():
        raise ValueError("Evaluation test must be nonempty with unique non-null IDs")
    allowed = range(7) if domain == "vsmec" else range(3)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if not labels.isin(allowed).all():
        raise ValueError("Invalid test labels")
    frame["label"] = labels.astype(int)
    return frame


def labels_for_test(frame, domain):
    original = frame["label"].to_numpy(dtype=int)
    return map_labels7(original) if domain == "vsmec" else original


def view_jobs(job):
    return {
        "zsA": [("V3-NEU", "sum"), ("V3-NEU", "argmax")],
        "zsB": [("V3-VSMEC", "none"), ("V3-VSMEC-strict", "none")],
        "refV": [("V7-VSMEC", "none"), ("V3-VSMEC", "sum"), ("V3-VSMEC", "argmax"), ("V3-VSMEC-strict", "sum"), ("V3-VSMEC-strict", "argmax")],
        "refN": [("V3-NEU", "none")],
    }[job]
