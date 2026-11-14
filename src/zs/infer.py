"""FP32 inference on frozen test rows only; no training or target adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml

from src.common.hashing import sha256_file
from src.common.schema import ZS_COLUMNS, ZS_PRED_COLUMNS
from src.data.preprocess import preprocess_text
from src.zs.ckpt import locate_checkpoint
from src.zs.mapping import project_probabilities, validate_probabilities
from src.zs.views import VIEW_LABELS, labels_for_test, load_test_frame, strict_mask, view_jobs

JOB_RUN_TAG = {"zsA": "ZSV2N", "zsB": "ZSN2V", "refV": "REFV2V", "refN": "REFN2N"}
JOB_SPECS = {"zsA": ("vsmec", "neuesc", 7), "zsB": ("neu3", "vsmec", 3),
             "refV": ("vsmec", "vsmec", 7), "refN": ("neu3", "neuesc", 3)}


def build_zs_run_id(alias, job, seed):
    if job not in JOB_RUN_TAG:
        raise ValueError(f"Unknown job: {job}")
    return f"{alias}_{JOB_RUN_TAG[job]}-fp32_s{int(seed)}"


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
    temp.replace(path)


def evaluation_data_hash(data_dir, target):
    """Step 1 hash semantics, without reading VSMEC train/dev data at inference."""
    directory = Path(data_dir)
    if target == "neuesc":
        return sha256_file(directory / "neuesc_100.csv")
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(directory / "vsmec_test.csv") != manifest.get("vsmec_test.csv"):
        raise ValueError("Evaluated VSMEC test does not match the frozen MANIFEST.json")
    combined = hashlib.sha256()
    for split in ("train", "dev", "test"):
        filename = f"vsmec_{split}.csv"
        digest = manifest.get(filename)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Missing/invalid frozen digest: {filename}")
        combined.update(f"{filename}:{digest}\n".encode("utf-8"))
    return combined.hexdigest()


def predict_checkpoint(checkpoint, texts, num_labels, batch_size, max_length):
    """Load locally saved tokenizer+encoder; restore original order after length sorting."""
    import torch
    from src.plm.model_ckpt import Stage1Classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = Stage1Classifier.load_stage1_checkpoint(checkpoint, device=device)
    if model.num_labels != num_labels:
        raise ValueError(f"Checkpoint head has {model.num_labels} labels, expected {num_labels}")
    model.float().eval()
    tokens = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    encodings = [{key: values[i] for key, values in tokens.items()} for i in range(len(texts))]
    order = sorted(range(len(texts)), key=lambda i: len(encodings[i]["input_ids"]), reverse=True)
    probabilities = np.empty((len(texts), num_labels), dtype=np.float64)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            padded = tokenizer.pad([encodings[i] for i in indices], padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in padded.items()}
            probabilities[indices] = torch.softmax(model(**batch).logits.float(), dim=1).cpu().numpy()
    return validate_probabilities(probabilities, num_labels)


def _prediction_frame(frame, target, num_labels, probabilities):
    result = pd.DataFrame({"id": frame["id"], "y_true3": labels_for_test(frame, target)})
    if target == "vsmec":
        result["y_true7"] = frame["label"].to_numpy(dtype=int)
        result["in_strict"] = strict_mask(result["y_true7"])
    else:
        result["y_true7"] = -1
        result["in_strict"] = np.nan
    if num_labels == 7:
        probs3, summed, mapped = project_probabilities(probabilities)
        result["y_pred3_sum"], result["y_pred3_argmax"] = summed, mapped
        result["y_pred7"] = np.argmax(probabilities, axis=1)
        for label in range(7):
            result[f"p7_{label}"] = probabilities[:, label]
    else:
        probs3 = probabilities
        result["y_pred3_sum"] = result["y_pred3_argmax"] = np.argmax(probs3, axis=1)
        result["y_pred7"] = -1
        for label in range(7):
            result[f"p7_{label}"] = np.nan
    for index, name in enumerate(("neg", "neu", "pos")):
        result[f"p3_{name}"] = probs3[:, index]
    return result.loc[:, list(ZS_PRED_COLUMNS)]


def score_prediction_view(prediction, label_view, map_method):
    data = prediction
    if label_view == "V3-VSMEC-strict":
        data = data.loc[strict_mask(data["y_true7"].to_numpy())]
    if data.empty:
        raise ValueError(f"No test examples in {label_view}")
    labels = VIEW_LABELS[label_view]
    y_true = data["y_true7" if label_view == "V7-VSMEC" else "y_true3"].to_numpy(dtype=int)
    column = "y_pred7" if label_view == "V7-VSMEC" else ("y_pred3_argmax" if map_method == "argmax" else "y_pred3_sum")
    y_pred = data[column].to_numpy(dtype=int)
    per_class = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    f1_slots = [np.nan] * (7 if label_view == "V7-VSMEC" else 3)
    for label, value in zip(labels, per_class):
        f1_slots[label] = float(value)
    return {
        "n_eval": len(data), "macro_f1": float(per_class.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)), "f1_per_class": json.dumps([None if np.isnan(x) else x for x in f1_slots]),
        **{f"f1_{name}": (f1_slots[i] if label_view != "V7-VSMEC" else np.nan) for i, name in enumerate(("neg", "neu", "pos"))},
    }


def run_job(config_path="configs/zs.yaml", backbone="PhoBERT", seed=0, job="zsA", *,
            vsmec_ckpt_root=None, neu_ckpt_root=None, data_dir="data/clean", results_dir="results",
            max_examples=None, batch_size=None, predictor=None):
    """Evaluate one job. Injected predictor(texts, num_labels, seed) returns probabilities."""
    started = time.monotonic()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if backbone not in config["backbones"]:
        raise ValueError(f"Unknown backbone alias: {backbone}")
    spec = config["backbones"][backbone]
    alias = spec.get("alias", backbone)
    src_domain, target, num_labels = JOB_SPECS[job]
    run_id = build_zs_run_id(alias, job, seed)
    size = int(batch_size or config.get("batch_size", 128))
    if size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_root = vsmec_ckpt_root if src_domain == "vsmec" else neu_ckpt_root
    if checkpoint_root is None:
        raise ValueError(f"Missing checkpoint root for {src_domain}")
    checkpoint = locate_checkpoint(checkpoint_root, alias, src_domain, seed)
    print(f"[Precision] FP32 | checkpoint={checkpoint}")
    frame = load_test_frame(data_dir, target, max_examples)
    data_hash = evaluation_data_hash(data_dir, target)
    out_dir = Path(results_dir)
    results_csv = out_dir / "zs.csv"
    eval_set = "vsmec_test" if target == "vsmec" else "neuesc_test"
    pred_path = out_dir / "predictions_zs" / f"{run_id}_{eval_set}.csv"
    required = set(view_jobs(job))
    existing = pd.DataFrame(columns=ZS_COLUMNS)
    if results_csv.is_file():
        existing = pd.read_csv(results_csv, keep_default_na=False)
        if tuple(existing.columns) != ZS_COLUMNS:
            raise ValueError("zs.csv has an incompatible column schema")
        if existing.duplicated(["run_id", "label_view", "map_method"]).any():
            raise ValueError("zs.csv contains duplicate keys")
        prior = existing.loc[existing["run_id"].eq(run_id)]
        if not prior.empty and set(prior["data_hash"].astype(str)) != {data_hash}:
            raise ValueError("Existing run uses different frozen data; use a separate results directory")
        complete = required.issubset(set(zip(prior["label_view"], prior["map_method"])))
        if complete and pred_path.is_file():
            previous = pd.read_csv(pred_path)
            if tuple(previous.columns) != ZS_PRED_COLUMNS or previous["id"].tolist() != frame["id"].tolist():
                raise ValueError("Existing run covers a different test population; use a separate results directory for smoke tests")
            print(f"=== [SKIP] run_id={run_id} ===")
            return {"skipped": True, "run_id": run_id, "prediction_path": pred_path, "results_csv": results_csv}
    column = spec.get("text_col", "text_seg" if alias == "PhoBERT" else "text")
    required_column = "text_seg" if alias == "PhoBERT" else "text"
    if column != required_column or spec.get("emoji_mode", "keep") != "keep":
        raise ValueError("Expected PhoBERT text_seg or ViSoBERT text, with emoji_mode=keep")
    texts = frame[column].astype(str).tolist()
    if any(not text.strip() for text in texts):
        raise ValueError("Empty test input text")
    if alias == "ViSoBERT":
        texts = [preprocess_text(text, emoji_mode="keep") for text in texts]
    probabilities = (predict_checkpoint(checkpoint, texts, num_labels, size, int(config.get("max_length", 256)))
                     if predictor is None else predictor(texts=texts, num_labels=num_labels, seed=int(seed)))
    probabilities = validate_probabilities(probabilities, num_labels)
    if len(probabilities) != len(frame):
        raise ValueError("Predictor returned the wrong number of test rows")
    predictions = _prediction_frame(frame, target, num_labels, probabilities)
    _atomic_csv(predictions, pred_path)
    predictions = pd.read_csv(pred_path)  # Metrics come from the exported artifact.
    src_name = "VSMEC" if src_domain == "vsmec" else "NEU"
    target_name = "VSMEC" if target == "vsmec" else "NEU"
    rows = []
    for label_view, method in view_jobs(job):
        rows.append({
            "run_id": run_id, "backbone": alias, "stage1": src_name, "stage2": "", "eval_domain": target_name,
            "label_view": label_view, "mode": "ref" if src_name == target_name else "zs", "fraction": 100,
            "seed": int(seed), "split": "test", "map_method": method,
            **score_prediction_view(predictions, label_view, method), "ckpt_path": checkpoint.as_posix(),
            "src_domain": src_name, "tgt_domain": target_name, "precision": "fp32", "code_version": _git_commit(),
            "data_hash": data_hash, "config_path": str(config_path),
        })
    retained = existing.loc[~existing["run_id"].eq(run_id)]
    combined = pd.concat([retained, pd.DataFrame(rows, columns=ZS_COLUMNS)], ignore_index=True)
    _atomic_csv(combined.loc[:, list(ZS_COLUMNS)], results_csv)
    print(f"=== [DONE] {run_id}: n={len(frame)}, macro_f1={rows[0]['macro_f1']:.6f}, elapsed={time.monotonic()-started:.2f}s ===")
    return {"skipped": False, "run_id": run_id, "rows": rows, "prediction_path": pred_path, "results_csv": results_csv}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", default="configs/zs.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job", choices=JOB_RUN_TAG, required=True)
    parser.add_argument("--vsmec-ckpt-root")
    parser.add_argument("--neu-ckpt-root")
    parser.add_argument("--data-dir", default="data/clean")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int)
    run_job(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
