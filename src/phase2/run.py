"""One Phase 2 run with content fingerprints, complete artifacts and safe reuse."""
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import yaml

from src.common.hashing import sha256_file
from src.phase2.data import load_phase2_prepared_data
from src.phase2.models import (artifact_path, build_phase2_model, count_model_parameters,
                               file_hashes, resolve_model_source)
from src.phase2.protocol import (build_phase2_run_id, canonical_digest,
    compute_dev_selection_score, compute_effective_cfg_hash8, load_phase2_config)
from src.phase2.trainer import Phase2Trainer
from src.zs.infer import _atomic_csv, _git_commit, _prediction_frame, score_prediction_view
from src.zs.mapping import validate_probabilities
from src.zs.views import view_jobs


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def hook_identity(fn):
    if fn is None:
        return None
    obj = fn if inspect.isfunction(fn) or inspect.isclass(fn) else type(fn)
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        raise ValueError("Custom hooks/trainers require inspectable source for provenance")
    return {"name": f"{obj.__module__}.{obj.__qualname__}", "source": canonical_digest(source)}


def score_view(prediction, label_view, map_method):
    if label_view != "V5-VSMEC":
        metrics = score_prediction_view(prediction, label_view, map_method)
        return {k: None if isinstance(v, float) and not math.isfinite(v) else v
                for k, v in metrics.items()}
    truth, pred = prediction.y_true7.to_numpy(int), prediction.y_pred7.to_numpy(int)
    per_class = f1_score(truth, pred, labels=list(range(5)), average=None, zero_division=0)
    return {"n_eval": len(prediction), "macro_f1": float(per_class.mean()),
            "weighted_f1": float(f1_score(truth, pred, labels=list(range(5)), average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(truth, pred)), "f1_per_class": json.dumps(per_class.tolist())}


def required_views(direction, run_mode):
    dev = [("V3-NEU", "none")] if direction == "SA" else [("V5-VSMEC", "none"), ("V7-VSMEC", "none")]
    result = [("dev", v, m) for v, m in dev]
    if run_mode == "main":
        test = view_jobs("refN" if direction == "SA" else "refV")
        if direction == "SB":
            test = test + [("V5-VSMEC", "none")]
        result += [("test", v, m) for v, m in test]
    return result


def run_phase2_experiment(config_path="configs/phase2.yaml", *, backbone="ViSoBERT",
    direction="SA", method="dft", fraction=10, seed=0, lr=2e-5, run_mode="main",
    clean_dir="data/clean", subsets_dir="data/subsets_phase2", results_dir="results/phase2",
    ckpt_out_dir="checkpoints/phase2", pretrained_model_path=None, source_ckpt_root=None,
    c0_stage1_run_dir=None, selection_path=None, max_examples=None, no_save_ckpt=False,
    strict_precision_override=None, epochs_override=None, method_tag_override=None,
    run_overrides=None, optimizer_builder=None, model_adapter_fn=None, checkpoint_saver=None, trainer=None):
    cfg = load_phase2_config(config_path)
    if direction not in cfg["directions"] or backbone not in ("ViSoBERT", "XLMR"):
        raise ValueError("Unsupported direction/backbone")
    base_method = "seq" if method.startswith("seq-") else method
    if base_method not in ("dft", "seq", "c0"):
        raise ValueError("Unsupported method")
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if method.startswith("seq-") and optimizer_builder is None and model_adapter_fn is None:
        raise ValueError("Extended method requires Task 10 hooks")
    settings = {**cfg["training"], **dict(run_overrides or {}), "lr": float(lr)}
    if strict_precision_override is not None:
        settings["strict_precision"] = bool(strict_precision_override)
    if epochs_override is not None:
        settings["epochs"] = int(epochs_override)
    hooks = {"optimizer": hook_identity(optimizer_builder), "adapter": hook_identity(model_adapter_fn),
             "saver": hook_identity(checkpoint_saver), "trainer": hook_identity(trainer)}
    overrides = {"settings": settings, "run_mode": run_mode, "max_examples": max_examples,
                 "save_ckpt": run_mode == "main" and not no_save_ckpt, "hooks": hooks}
    run_id = build_phase2_run_id(backbone, direction, method_tag_override or method,
                                fraction, seed, lr, cfg, run_overrides=overrides)
    data = load_phase2_prepared_data(direction, fraction, seed, backbone, cfg, run_mode=run_mode,
                                    clean_dir=clean_dir, subsets_dir=subsets_dir, max_examples=max_examples)
    source_args = dict(pretrained_model_path=pretrained_model_path, source_ckpt_root=source_ckpt_root,
                       c0_stage1_run_dir=c0_stage1_run_dir, expected_ids_sha256=data["provenance"]["ids_sha256"],
                       results_dir=results_dir)
    _, source = resolve_model_source(base_method, direction, backbone, seed, fraction, cfg, **source_args)
    if base_method == "c0" and source["stage1_source_files_sha256"] != data["provenance"]["source_files_sha256"]:
        raise ValueError("C0 low-resource violation: source data hashes mismatch")
    selection = None
    if selection_path is not None:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        selected_method = "seq" if base_method == "c0" else base_method
        selected = selection["selected"][f"{backbone}/{direction}/{selected_method}"]
        if (selection.get("protocol_hash") != canonical_digest(cfg) or float(selected["lr"]) != float(lr)
            or selected["data_hash"] != canonical_digest(data["provenance"]["source_files_sha256"])):
            raise ValueError("Locked LR selection does not match protocol/LR/data")
    commit = _git_commit()
    payload = {"protocol_cfg": cfg, "run_overrides": dict(run_overrides or {}), "effective": overrides,
        "method": method, "direction": direction, "fraction": int(fraction), "seed": int(seed),
        "backbone": backbone, "lr": float(lr), "data_provenance": data["provenance"],
        "source_ckpt_provenance": source, "git_commit": commit, "selection": selection,
        "source_code_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "ckpt_out_dir": str(Path(ckpt_out_dir).resolve())}
    fingerprint = canonical_digest(payload)
    result_root = Path(results_dir).resolve()
    run_dir, work_dir = result_root / "runs" / run_id, result_root / "work" / run_id
    checkpoint = Path(ckpt_out_dir).resolve() / run_id
    completion_path = run_dir / "completion.json"
    csv_path = result_root / "phase2_runs.csv"
    existing = pd.read_csv(csv_path, keep_default_na=False) if csv_path.exists() else pd.DataFrame()
    prior = existing.loc[existing.run_id.eq(run_id)] if not existing.empty else existing
    views = required_views(direction, run_mode)
    required = {"config.yaml", "metadata.json", "metrics.json", "predictions_dev.csv", "cost.json"}
    if run_mode == "main":
        required.add("predictions_test.csv")
    save_ckpt = overrides["save_ckpt"]
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("fingerprint") != fingerprint:
            raise ValueError("Completion fingerprint mismatch; use separate results directory")
        artifacts = completion.get("artifacts", {})
        valid = completion.get("complete") is True and required.issubset(artifacts)
        valid = valid and all(artifact_path(run_dir, p).is_file() and sha256_file(artifact_path(run_dir, p)) == h
                              for p, h in artifacts.items())
        if save_ckpt:
            ckpt_files = file_hashes(checkpoint)
            valid = valid and bool(ckpt_files) and bool(completion.get("ckpt_path"))
            valid = valid and set(ckpt_files) == set(completion.get("checkpoint_files", []))
            valid = valid and all(os.path.relpath(checkpoint / p, run_dir).replace("\\", "/") in artifacts for p in ckpt_files)
            if checkpoint_saver is None:
                valid = valid and "head.pt" in ckpt_files and any(p.startswith("encoder/") for p in ckpt_files)
        valid = valid and len(prior) == len(views) and set(zip(prior.get("split", []), prior.get("label_view", []), prior.get("map_method", []))) == set(views)
        # Detect changes to aggregate rows as well as per-run files.
        valid = valid and canonical_digest(prior.to_csv(index=False)) == completion.get("rows_digest")
        if valid:
            return dict(run_id=run_id, run_dir=run_dir, manifest=completion_path, skipped=True, rows=prior.to_dict("records"))
    provenance_file = work_dir / "provenance.json"
    if provenance_file.exists() and json.loads(provenance_file.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
        raise ValueError("Resume provenance fingerprint mismatch")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(provenance_file, {"fingerprint": fingerprint, **payload})
    # Invalidate previous completion before any output mutation.
    atomic_json(completion_path, {"complete": False, "fingerprint": fingerprint})
    init_provenance = {**source, "head_seed": int(seed)}
    if trainer is None:
        model, tokenizer, init_provenance = build_phase2_model(base_method, direction, backbone, seed,
            fraction, cfg, **source_args, model_adapter_fn=model_adapter_fn)
        trainer = Phase2Trainer(model, tokenizer, data["target_domain"],
            cfg["directions"][direction]["target_num_labels"], seed, settings,
            optimizer_builder=optimizer_builder, checkpoint_saver=checkpoint_saver)
    fit = trainer.fit(data["train_texts"], data["train_labels"], data["dev_texts"], data["dev_labels"], work_dir)
    metadata = {"run_id": run_id, "protocol_version": cfg["protocol_version"],
        "cfg_hash8": compute_effective_cfg_hash8(cfg, overrides), "backbone": backbone,
        "direction": direction, "method": method, "fraction": int(fraction), "seed": int(seed),
        "lr": float(lr), "run_mode": run_mode, "max_examples": max_examples,
        "data_provenance": data["provenance"], "source_ckpt_provenance": init_provenance,
        "git_commit": commit, "fingerprint": fingerprint}
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"protocol_cfg": cfg, "run": metadata,
                                                        "settings": settings}, sort_keys=True), encoding="utf-8")
    atomic_json(run_dir / "metadata.json", metadata)
    predictions, scores, rows = {}, {"dev": {}, "test": None}, []
    labels = cfg["directions"][direction]["target_num_labels"]
    for split in (("dev", "test") if run_mode == "main" else ("dev",)):
        probabilities = validate_probabilities(trainer.predict(data[f"{split}_texts"]), labels)
        frame = _prediction_frame(data[f"{split}_df"], data["target_domain"], labels, probabilities)
        path = run_dir / f"predictions_{split}.csv"
        _atomic_csv(frame, path)
        predictions[split] = pd.read_csv(path)
        if split == "test":
            scores[split] = {}
    dev_score = compute_dev_selection_score(data["dev_labels"],
        predictions["dev"]["y_pred7" if labels == 7 else "y_pred3_sum"].to_numpy(), data["target_domain"])
    for split, view, mapping in views:
        metrics = score_view(predictions[split], view, mapping)
        scores[split][f"{view}/{mapping}"] = metrics
        rows.append({"run_id": run_id, "backbone": backbone, "direction": direction, "method": method,
            "fraction": fraction, "seed": seed, "lr": lr, "run_mode": run_mode,
            "max_examples": max_examples, "protocol_version": cfg["protocol_version"],
            "data_hash": canonical_digest(data["provenance"]["source_files_sha256"]),
            "ids_sha256": data["provenance"]["ids_sha256"], "git_commit": commit,
            "split": split, "label_view": view, "map_method": mapping,
            "dev_selection_score": dev_score, **metrics})
    metrics = {k: fit.get(k) for k in ("history", "best_epoch", "best_step", "early_stopped", "actual_update_steps")}
    metrics.update(dev_selection_score=dev_score, dev_metrics=scores["dev"], test_metrics=scores["test"])
    atomic_json(run_dir / "metrics.json", metrics)
    cost = dict(fit.get("cost", getattr(trainer, "cost", {})))
    if hasattr(trainer, "model"):
        cost.update(count_model_parameters(trainer.model))
    for key in ("total_params", "trainable_params", "train_seconds", "gpu_hours", "peak_vram_mb",
                "gpu_name", "n_gpus", "trunc_rate", "actual_update_steps"):
        cost.setdefault(key, fit.get(key))
    if base_method == "c0":
        cost["stage1_dft_cost_ref"] = source["stage1_cost_ref"]
    atomic_json(run_dir / "cost.json", cost)
    if save_ckpt:
        trainer.save(checkpoint)
        if not (checkpoint / "head.pt").is_file() or not (checkpoint / "encoder").is_dir():
            if checkpoint_saver is None:
                raise ValueError("Missing encoder/head checkpoint")
    if not existing.empty:
        existing = existing.loc[~existing.run_id.eq(run_id)]
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    _atomic_csv(combined, csv_path)
    # Hash rows as read from disk to normalize CSV numeric/string representations.
    saved_rows = pd.read_csv(csv_path, keep_default_na=False)
    saved_rows = saved_rows.loc[saved_rows.run_id.eq(run_id)]
    files = [run_dir / name for name in sorted(required)]
    if save_ckpt:
        files += [checkpoint / name for name in file_hashes(checkpoint)]
    artifacts = {os.path.relpath(p, run_dir).replace("\\", "/"): sha256_file(p) for p in files}
    atomic_json(completion_path, {"complete": True, "fingerprint": fingerprint, "artifacts": artifacts,
        "ckpt_path": os.path.relpath(checkpoint, run_dir).replace("\\", "/") if save_ckpt else None,
        "checkpoint_files": sorted(file_hashes(checkpoint)) if save_ckpt else [],
        "rows_digest": canonical_digest(saved_rows.to_csv(index=False))})
    return dict(run_id=run_id, run_dir=run_dir, manifest=completion_path, skipped=False, rows=rows)
