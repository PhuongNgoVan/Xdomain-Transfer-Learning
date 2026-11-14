"""Orchestrate target-only selection followed by delayed test reporting."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.common.hashing import sha256_file
from src.common.schema import SEQ_COLUMNS
from src.data.preprocess import preprocess_text
from src.seq.protocol import DIRECTIONS, atomic_json, digest, load_config
from src.seq.trainer import TorchTrainer, load_encoder_new_head
from src.zs.ckpt import locate_checkpoint
from src.zs.infer import _atomic_csv, _git_commit, _prediction_frame, score_prediction_view
from src.zs.mapping import validate_probabilities
from src.zs.views import view_jobs

SEQ_TAGS = {"SA": "SEQV2N", "SB": "SEQN2V", "C0N": "C0N", "C0V": "C0V"}


def build_seq_run_id(alias, direction, variant, seed, lr):
    if not variant or "_" in variant or any(c in variant for c in "/\\"):
        raise ValueError("variant must be nonempty without underscores or path separators")
    return f"{alias}_{SEQ_TAGS[direction]}-{variant}_s{int(seed)}_lr{float(lr):g}"


def _load_split(directory, target, split, limit):
    filename = "neuesc_100.csv" if target == "neuesc" else f"vsmec_{split}.csv"
    frame = pd.read_csv(Path(directory) / filename, keep_default_na=False)
    frame = frame.loc[frame["subset"].eq(split)].copy().reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("max_examples must be positive")
        frame = frame.iloc[:limit].copy()
    allowed = range(3 if target == "neuesc" else 7)
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any() or not frame.label.isin(allowed).all():
        raise ValueError(f"Invalid target {split} rows")
    return frame


def _texts(frame, backbone, cfg):
    spec = cfg["backbones"][backbone]
    required = "text_seg" if backbone == "PhoBERT" else "text"
    if spec["text_col"] != required or spec["emoji_mode"] != "keep":
        raise ValueError("Preprocessing must match the frozen backbone protocol")
    values = frame[required].astype(str).tolist()
    if any(not s.strip() for s in values):
        raise ValueError("Empty target text")
    return [preprocess_text(x, emoji_mode="keep") for x in values] if backbone == "ViSoBERT" else values


def run_stage2(config_path="configs/seq.yaml", backbone="PhoBERT", seed=0, direction="SA", lr=2e-5,
               variant="seq-bf16", *, vsmec_ckpt_root=None, neu_ckpt_root=None, data_dir="data/clean",
               results_dir="results", ckpt_out_dir="checkpoints/seq", max_examples=None,
               no_save_ckpt=False, trainer=None):
    started = time.monotonic()
    cfg = load_config(config_path)
    source, target, n_labels = DIRECTIONS[direction]
    sweep = variant.startswith("seqsweep-")
    if sweep and (seed != cfg["sweep_seed"] or float(lr) not in cfg["lr_grid"]):
        raise ValueError("Sweep requires seed0 and an LR from the frozen grid")
    if not sweep and seed not in cfg["c0_seeds" if direction.startswith("C0") else "seeds"]:
        raise ValueError("Seed outside configured protocol")
    if not sweep:
        selection_path = Path(results_dir) / "selection_seq.json"
        if not selection_path.exists():
            raise ValueError("Run complete dev sweeps and select_lr before main/C0 runs")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_direction = {"C0N": "SA", "C0V": "SB"}.get(direction, direction)
        if selection.get("protocol_hash") != digest(cfg["training"]) or float(selection["selected"][f"{backbone}/{selection_direction}"]["lr"]) != float(lr):
            raise ValueError("Main/C0 LR or protocol differs from dev selection")
    checkpoint = locate_checkpoint(vsmec_ckpt_root if source == "vsmec" else neu_ckpt_root, backbone, source, seed)
    run_id = build_seq_run_id(backbone, direction, variant, seed, lr)
    root, destination = Path(results_dir), Path(ckpt_out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    settings = {**cfg["training"], "lr": float(lr)}
    files = ["neuesc_100.csv"] if target == "neuesc" else [f"vsmec_{s}.csv" for s in (("train", "dev") if sweep else ("train", "dev", "test"))]
    hashes = {f: sha256_file(Path(data_dir) / f) for f in files}
    data_hash = digest(hashes)
    code_root = Path(__file__).resolve().parents[1]
    fingerprint = digest({"config": cfg, "data": hashes, "source": {str(p.relative_to(checkpoint)): sha256_file(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
                          "seed": seed, "lr": lr, "direction": direction, "variant": variant, "max_examples": max_examples,
                          "save": not no_save_ckpt and not sweep, "code_version": _git_commit(),
                          "source_code": {p.relative_to(code_root).as_posix(): sha256_file(p) for p in sorted(code_root.rglob("*.py"))}})
    work = root / "work_seq" / run_id
    manifest_path = root / "runs_seq" / f"{run_id}.json"
    results_csv = root / "seq.csv"
    old = pd.read_csv(results_csv, keep_default_na=False) if results_csv.exists() else pd.DataFrame(columns=SEQ_COLUMNS)
    if tuple(old.columns) != SEQ_COLUMNS or old.duplicated(["run_id", "split", "label_view", "map_method"]).any():
        raise ValueError("Invalid seq.csv schema/duplicate keys")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Run inputs/config changed; use a new results directory or variant")
        rows = old.loc[old.run_id.eq(run_id)]
        main_view = "V3-NEU" if n_labels == 3 else "V7-VSMEC"
        required = {("dev", main_view, "none")}
        if not sweep:
            required.update(("test", view, method) for view, method in view_jobs("refN" if n_labels == 3 else "refV"))
        actual_keys = set(zip(rows.split, rows.label_view, rows.map_method))
        if manifest.get("complete") and len(rows) == manifest["n_rows"] and actual_keys == required and rows.data_hash.eq(data_hash).all():
            artifacts_ok = all(Path(p).is_file() and sha256_file(p) == h for p, h in manifest["artifacts"].items())
            if artifacts_ok:
                print(f"=== [SKIP] run_id={run_id} ===")
                return {"skipped": True, "run_id": run_id}
    atomic_json(manifest_path, {"fingerprint": fingerprint, "complete": False})
    work_provenance = work / "provenance.json"
    if work_provenance.exists():
        if json.loads(work_provenance.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Resume state belongs to different inputs/protocol/code")
    elif (work / "resume.pt").exists():
        raise ValueError("Resume state is missing provenance; cannot safely reuse it")
    atomic_json(work_provenance, {"fingerprint": fingerprint})
    model = trainer if trainer is not None else TorchTrainer(checkpoint, n_labels, seed, settings)
    train = _load_split(data_dir, target, "train", max_examples)
    dev = _load_split(data_dir, target, "dev", max_examples)
    print(f"[Source] {checkpoint} | head_seed={seed} | lr={lr}")
    fitted = model.fit(_texts(train, backbone, cfg), train.label.astype(int).tolist(),
                       _texts(dev, backbone, cfg), dev.label.astype(int).tolist(), work)
    # Test is first opened/processed only after fit has restored the best dev model.
    splits = [("dev", dev)]
    if not sweep:
        splits.append(("test", _load_split(data_dir, target, "test", max_examples)))
    save = not sweep and not no_save_ckpt
    artifacts, rows = {}, []
    if save:
        model.save(destination)
        for p in destination.rglob("*"):
            if p.is_file():
                artifacts[str(p.resolve())] = sha256_file(p)
    for split, frame in splits:
        probs = validate_probabilities(model.predict(_texts(frame, backbone, cfg)), n_labels)
        if len(probs) != len(frame):
            raise ValueError("Prediction row count does not match target split")
        path = root / "predictions_seq" / f"{run_id}_{'neuesc' if target == 'neuesc' else 'vsmec'}_{split}.csv"
        _atomic_csv(_prediction_frame(frame, target, n_labels, probs), path)
        artifacts[str(path.resolve())] = sha256_file(path)
        pred = pd.read_csv(path)
        views = [("V3-NEU" if n_labels == 3 else "V7-VSMEC", "none")] if split == "dev" or n_labels == 3 else view_jobs("refV")
        for view, method in views:
            rows.append(dict(run_id=run_id, backbone=backbone, direction=direction, stage1="VSMEC" if source == "vsmec" else "NEU",
                             stage2="VSMEC" if target == "vsmec" else "NEU", eval_domain="VSMEC" if target == "vsmec" else "NEU",
                             label_view=view, mode="seq", fraction=100, seed=seed, split=split, map_method=method,
                             **score_prediction_view(pred, view, method), lr=lr, variant=variant, best_epoch=fitted["best_epoch"], best_step=fitted["best_step"],
                             n_train=len(train), trunc_rate=fitted.get("trunc_rate", 0.0), train_seconds=time.monotonic()-started,
                             src_ckpt_path=str(checkpoint), ckpt_path=str(destination.resolve()) if save else "", precision=settings["precision"],
                             code_version=_git_commit(), data_hash=data_hash, config_path=str(config_path)))
    _atomic_csv(pd.concat([old.loc[~old.run_id.eq(run_id)], pd.DataFrame(rows, columns=SEQ_COLUMNS)], ignore_index=True), results_csv)
    atomic_json(manifest_path, {"fingerprint": fingerprint, "complete": True, "n_rows": len(rows), "artifacts": artifacts,
                               "training": settings, "history": fitted.get("history", []), "data_hashes": hashes,
                               "protocol_hash": digest(cfg["training"]), "baseline_provenance": cfg["baseline_provenance"]})
    test_summary = "NA" if sweep else f"{next(r['macro_f1'] for r in rows if r['split'] == 'test'):.6f}"
    print(f"=== [DONE] {run_id}: n_test={0 if sweep else len(splits[-1][1])}, test_macro_f1={test_summary}, dev_macro_f1={rows[0]['macro_f1']:.6f}, best_step={fitted['best_step']}, elapsed={time.monotonic()-started:.2f}s ===")
    return {"skipped": False, "run_id": run_id, "rows": rows, "manifest": manifest_path}
