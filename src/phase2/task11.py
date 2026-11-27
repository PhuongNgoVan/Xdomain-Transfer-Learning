"""Task 11 utilities: XLMR zero-shot evaluation, unified encoder benchmarking, and summary reporting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common.constants import EMOTIONS7, ID_TO_LABEL3
from src.common.hashing import sha256_file
from src.data.preprocess import preprocess_text
from src.phase2.audit_task01_02 import check_underscore_leakage
from src.phase2.models import _locate_phase2_xlmr_source_checkpoint, artifact_path, count_model_parameters
from src.phase2.protocol import apply_phase2_holm, canonical_digest, load_phase2_config
from src.phase2.run import atomic_json, score_view
from src.seq.analysis import paired_gain
from src.zs.infer import (
    _atomic_csv,
    _git_commit,
    _prediction_frame,
    evaluation_data_hash,
    predict_checkpoint,
)
from src.zs.mapping import validate_probabilities
from src.zs.views import VIEW_LABELS, load_test_frame, strict_mask, view_jobs

PRETRAINING_LEAKAGE_NOTE = (
    "Public Vietnamese datasets (UIT-VSMEC, NEU-ESC) may overlap with pretraining "
    "corpora of open-weight LLMs; results are an upper-bound reference rather than "
    "strictly unseen zero-shot generalization."
)

_ZS_REQUIRED_ARTIFACTS = ("predictions_test.csv", "metrics.json", "metadata.json", "cost.json")


def _extract_view_arrays(pred: pd.DataFrame, view: str, mapping: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    if view == "V3-VSMEC-strict":
        sub = pred.loc[strict_mask(pred["y_true7"].to_numpy())].reset_index(drop=True)
        col = "y_pred3_argmax" if mapping == "argmax" else "y_pred3_sum"
        return sub["id"].to_numpy(), sub["y_true3"].to_numpy(dtype=int), sub[col].to_numpy(dtype=int), (0, 2)
    if view == "V7-VSMEC":
        return pred["id"].to_numpy(), pred["y_true7"].to_numpy(dtype=int), pred["y_pred7"].to_numpy(dtype=int), tuple(range(7))
    if view == "V5-VSMEC":
        return pred["id"].to_numpy(), pred["y_true7"].to_numpy(dtype=int), pred["y_pred7"].to_numpy(dtype=int), tuple(range(5))
    if view in ("V3-NEU", "V3-VSMEC"):
        col = "y_pred3_argmax" if mapping == "argmax" else "y_pred3_sum"
        return pred["id"].to_numpy(), pred["y_true3"].to_numpy(dtype=int), pred[col].to_numpy(dtype=int), (0, 1, 2)
    raise ValueError(f"Unsupported view: {view}")


def run_xlmr_zs_eval(
    results_dir: str | Path = "results/phase2",
    ckpt_dir: str | Path = "checkpoints/phase2",
    *,
    config_path: str | Path = "configs/phase2.yaml",
    clean_dir: str | Path = "data/clean",
    out_dir: Optional[str | Path] = None,
    directions: Sequence[str] = ("SA", "SB"),
    seeds: Sequence[int] = (0, 1, 2),
    batch_size: int = 32,
    max_examples: Optional[int] = None,
    predictor_fn: Optional[Callable[..., np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run FP32 zero-shot evaluation for XLMR source checkpoints with completion/SKIP verification."""
    cfg = load_phase2_config(config_path)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and int(max_examples) <= 0:
        raise ValueError("max_examples must be positive")

    out_root = Path(out_dir).resolve() if out_dir is not None else (Path(results_dir).resolve() / "xlmr_zs")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "xlmr_zs_runs.csv"
    existing = pd.read_csv(csv_path, keep_default_na=False) if csv_path.is_file() else pd.DataFrame()

    commit = _git_commit()
    code_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    max_length = int(cfg["training"]["max_length"])

    run_records: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for direction in directions:
        if direction not in cfg["directions"]:
            raise ValueError(f"Unsupported direction: {direction}")
        job = "zsA" if direction == "SA" else "zsB"
        target = cfg["directions"][direction]["target"]
        src_labels = int(cfg["directions"][direction]["source_num_labels"])
        views = view_jobs(job)

        test_df = load_test_frame(clean_dir, target, max_examples=max_examples)
        data_hash = evaluation_data_hash(clean_dir, target)
        raw = test_df["text"].astype(str).tolist()
        texts = [preprocess_text(t, emoji_mode="keep") for t in raw]
        if not texts or any(not t.strip() for t in texts):
            raise ValueError(f"Empty test input text for {direction}")
        if check_underscore_leakage(raw, texts)["status"] != "PASS":
            raise ValueError(f"Underscore leakage in {direction} ZS test inputs")
        test_ids_sha = canonical_digest(test_df["id"].tolist())

        for seed in seeds:
            ckpt, xlmr_meta = _locate_phase2_xlmr_source_checkpoint(
                ckpt_dir,
                direction,
                int(seed),
                results_dir=results_dir,
                protocol_cfg=cfg,
                return_meta=True,
            )
            from src.plm.model_ckpt import _find_encoder_weight_file

            enc_sha = sha256_file(_find_encoder_weight_file(ckpt / "encoder"))
            head_sha = sha256_file(ckpt / "head.pt")
            run_overrides = {
                "batch_size": int(batch_size),
                "max_examples": max_examples,
                "max_length": max_length,
            }
            cfg_hash8 = canonical_digest(
                {
                    "protocol_version": cfg["protocol_version"],
                    "direction": direction,
                    "method": "zs",
                    "seed": int(seed),
                    **run_overrides,
                }
            )[:8]
            run_id = f"XLMR_{direction}_zs_s{int(seed)}_{cfg['protocol_tag']}_h{cfg_hash8}"
            payload = {
                "protocol_cfg": cfg,
                "backbone": "XLMR",
                "direction": direction,
                "method": "zs",
                "seed": int(seed),
                "target": target,
                "job": job,
                "data_hash": data_hash,
                "test_ids_sha256": test_ids_sha,
                "source_ckpt_run_id": xlmr_meta["run_id"],
                "encoder_sha256": enc_sha,
                "head_sha256": head_sha,
                "run_overrides": run_overrides,
                "git_commit": commit,
                "source_code_sha256": code_hashes,
            }
            fingerprint = canonical_digest(payload)

            run_dir = out_root / "runs" / run_id
            work_dir = out_root / "work" / run_id
            completion_path = run_dir / "completion.json"
            prior = existing.loc[existing["run_id"].eq(run_id)] if not existing.empty else existing

            if completion_path.is_file():
                comp = json.loads(completion_path.read_text(encoding="utf-8"))
                if comp.get("fingerprint") != fingerprint:
                    raise ValueError(f"XLMR ZS completion fingerprint mismatch for {run_id}")
                artifacts = comp.get("artifacts", {})
                valid = comp.get("complete") is True and set(_ZS_REQUIRED_ARTIFACTS).issubset(artifacts)
                valid = valid and all(
                    artifact_path(run_dir, rel).is_file()
                    and sha256_file(artifact_path(run_dir, rel)) == digest
                    for rel, digest in artifacts.items()
                )
                valid = valid and (
                    len(prior) == len(views)
                    and set(zip(prior.get("label_view", []), prior.get("map_method", []))) == set(views)
                    and canonical_digest(prior.to_csv(index=False)) == comp.get("rows_digest")
                )
                if valid:
                    run_records.append({"run_id": run_id, "run_dir": run_dir, "skipped": True})
                    all_rows.extend(prior.to_dict("records"))
                    continue

            run_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(work_dir / "provenance.json", {"fingerprint": fingerprint, **payload})
            atomic_json(completion_path, {"complete": False, "fingerprint": fingerprint})

            t0 = time.perf_counter()
            if predictor_fn is not None:
                probs = predictor_fn(
                    ckpt=ckpt,
                    texts=texts,
                    num_labels=src_labels,
                    batch_size=int(batch_size),
                    max_length=max_length,
                )
            else:
                probs = predict_checkpoint(ckpt, texts, src_labels, int(batch_size), max_length)
            elapsed = time.perf_counter() - t0

            probs = validate_probabilities(probs, src_labels)
            if len(probs) != len(test_df):
                raise ValueError(f"Prediction row count mismatch: {len(probs)} != {len(test_df)}")
            pred_frame = _prediction_frame(test_df, target, src_labels, probs)
            pred_path = run_dir / "predictions_test.csv"
            _atomic_csv(pred_frame, pred_path)
            pred_read = pd.read_csv(pred_path)

            view_metrics: Dict[str, Any] = {}
            rows: List[Dict[str, Any]] = []
            for view, map_method in views:
                metrics = score_view(pred_read, view, map_method)
                view_metrics[f"{view}/{map_method}"] = metrics
                rows.append(
                    {
                        "run_id": run_id,
                        "backbone": "XLMR",
                        "direction": direction,
                        "method": "zs",
                        "fraction": 100,
                        "seed": int(seed),
                        "protocol_version": cfg["protocol_version"],
                        "max_examples": max_examples,
                        "data_hash": data_hash,
                        "git_commit": commit,
                        "split": "test",
                        "label_view": view,
                        "map_method": map_method,
                        **metrics,
                    }
                )

            atomic_json(
                run_dir / "metrics.json",
                {
                    "run_id": run_id,
                    "direction": direction,
                    "method": "zs",
                    "seed": int(seed),
                    "n_eval": int(len(pred_read)),
                    "test_metrics": view_metrics,
                },
            )
            atomic_json(
                run_dir / "metadata.json",
                {
                    "run_id": run_id,
                    "protocol_version": cfg["protocol_version"],
                    "backbone": "XLMR",
                    "direction": direction,
                    "method": "zs",
                    "job": job,
                    "seed": int(seed),
                    "max_examples": max_examples,
                    "data_hash": data_hash,
                    "test_ids_sha256": test_ids_sha,
                    "source_ckpt_provenance": {
                        "src_ckpt_path": str(ckpt),
                        "source_run_origin": "phase2_xlmr_dft_100pct",
                        "source_dft_run_id": xlmr_meta["run_id"],
                        "source_dft_verified": xlmr_meta["verified_completion"],
                        "encoder_sha256": enc_sha,
                        "head_sha256": head_sha,
                    },
                    "git_commit": commit,
                    "fingerprint": fingerprint,
                },
            )
            n_eval = len(texts)
            atomic_json(
                run_dir / "cost.json",
                {
                    "precision": "fp32",
                    "batch_size": int(batch_size),
                    "max_length": max_length,
                    "n_eval": int(n_eval),
                    "elapsed_seconds": float(elapsed),
                    "throughput_samples_per_sec": float(n_eval / elapsed) if elapsed > 0 else 0.0,
                    "gpu_hours_per_1000_samples": float((elapsed / n_eval) * 1000.0 / 3600.0) if (elapsed > 0 and n_eval > 0) else 0.0,
                },
            )

            if not existing.empty:
                existing = existing.loc[~existing["run_id"].eq(run_id)]
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            _atomic_csv(existing, csv_path)
            saved_rows = pd.read_csv(csv_path, keep_default_na=False)
            saved_rows = saved_rows.loc[saved_rows["run_id"].eq(run_id)]
            artifacts = {name: sha256_file(run_dir / name) for name in sorted(_ZS_REQUIRED_ARTIFACTS)}
            atomic_json(
                completion_path,
                {
                    "complete": True,
                    "fingerprint": fingerprint,
                    "artifacts": artifacts,
                    "rows_digest": canonical_digest(saved_rows.to_csv(index=False)),
                },
            )
            run_records.append({"run_id": run_id, "run_dir": run_dir, "skipped": False})
            all_rows.extend(rows)

    return {"out_dir": out_root, "csv_path": csv_path, "runs": run_records, "rows": all_rows}


def benchmark_encoder_inference(
    backbones: Sequence[str] = ("ViSoBERT", "XLMR"),
    domains: Sequence[str] = ("neuesc", "vsmec"),
    *,
    config_path: str | Path = "configs/phase2.yaml",
    clean_dir: str | Path = "data/clean",
    precision: str = "fp32",
    n_warmup: int = 3,
    n_b1: int = 100,
    large_batch_size: int = 32,
    n_large: Optional[int] = None,
    pretrained_paths: Optional[Mapping[str, str | Path]] = None,
    local_files_only: bool = False,
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Benchmark ViSoBERT and XLMR inference latency, throughput, VRAM, and params on identical hardware."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src.plm.model_ckpt import Stage1Classifier

    if precision not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported encoder benchmark precision: {precision}")
    cfg = load_phase2_config(config_path)
    max_length = int(cfg["training"]["max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16 encoder benchmark on GPU without bf16 support")

    domain_texts: Dict[str, List[str]] = {}
    for domain in domains:
        test_df = load_test_frame(clean_dir, domain)
        raw = test_df["text"].astype(str).tolist()
        texts = [preprocess_text(t, emoji_mode="keep") for t in raw]
        if check_underscore_leakage(raw, texts)["status"] != "PASS":
            raise ValueError(f"Underscore leakage in {domain} benchmark inputs")
        domain_texts[domain] = texts

    benchmarks: List[Dict[str, Any]] = []
    for backbone in backbones:
        if backbone not in ("ViSoBERT", "XLMR"):
            raise ValueError(f"Unsupported backbone for benchmark: {backbone}")
        model_ref = (
            str(pretrained_paths[backbone])
            if pretrained_paths and backbone in pretrained_paths
            else str(cfg["backbones"][backbone]["model_name"])
        )
        tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_files_only)

        for domain in domains:
            num_labels = 3 if domain == "neuesc" else 7
            texts = domain_texts[domain]
            encoder = AutoModel.from_pretrained(model_ref, local_files_only=local_files_only)
            model = Stage1Classifier(encoder, num_labels, dropout=float(cfg["training"]["dropout"]))
            param_counts = count_model_parameters(model)
            model = model.to(device)
            if use_bf16:
                model = model.bfloat16()
            else:
                model = model.float()
            model.eval()

            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            for _ in range(max(1, int(n_warmup))):
                enc = tokenizer(texts[:1], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                with torch.inference_mode():
                    model(**{k: v.to(device) for k, v in enc.items()})
            if device.type == "cuda":
                torch.cuda.synchronize()

            b1_subset = texts[: min(int(n_b1), len(texts))]
            b1_times: List[float] = []
            for txt in b1_subset:
                enc = tokenizer([txt], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model(**{k: v.to(device) for k, v in enc.items()})
                if device.type == "cuda":
                    torch.cuda.synchronize()
                b1_times.append(time.perf_counter() - t0)

            batch_texts = texts if n_large is None else texts[: min(int(n_large), len(texts))]
            encoded = tokenizer(batch_texts, padding=False, truncation=True, max_length=max_length)
            encodings = [{k: v[i] for k, v in encoded.items()} for i in range(len(batch_texts))]
            order = sorted(range(len(batch_texts)), key=lambda i: len(encodings[i]["input_ids"]), reverse=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, len(order), int(large_batch_size)):
                    idxs = order[start : start + int(large_batch_size)]
                    padded = tokenizer.pad([encodings[i] for i in idxs], padding=True, return_tensors="pt")
                    model(**{k: v.to(device) for k, v in padded.items()})
            if device.type == "cuda":
                torch.cuda.synchronize()
            large_elapsed = time.perf_counter() - t0

            n_proc = len(batch_texts)
            throughput = n_proc / large_elapsed if large_elapsed > 0 else 0.0
            gpu_h_1k = (large_elapsed / n_proc) * 1000.0 / 3600.0 if (n_proc > 0 and device.type == "cuda") else 0.0
            peak_vram = float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

            benchmarks.append(
                {
                    "backbone": backbone,
                    "domain": domain,
                    "num_labels": int(num_labels),
                    "model_ref": model_ref,
                    "precision": "bf16" if use_bf16 else "fp32",
                    "device": device.type,
                    "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                    "max_length": max_length,
                    "total_params": int(param_counts["total_params"]),
                    "finetune_trainable_params": int(param_counts["trainable_params"]),
                    "infer_trainable_params": 0,
                    "peak_vram_mb": peak_vram,
                    "n_b1": len(b1_times),
                    "latency_batch1_mean_ms": float(np.mean(b1_times) * 1000.0) if b1_times else 0.0,
                    "latency_batch1_p50_ms": float(np.median(b1_times) * 1000.0) if b1_times else 0.0,
                    "latency_batch1_p95_ms": float(np.percentile(b1_times, 95) * 1000.0) if b1_times else 0.0,
                    "large_batch_size": int(large_batch_size),
                    "n_large": int(n_proc),
                    "large_batch_seconds": float(large_elapsed),
                    "throughput_samples_per_sec": float(throughput),
                    "gpu_hours_per_1000_samples": float(gpu_h_1k),
                }
            )
            del model, encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "protocol_version": cfg["protocol_version"],
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "precision": "bf16" if use_bf16 else "fp32",
        "max_length": max_length,
        "large_batch_size": int(large_batch_size),
        "benchmarks": benchmarks,
    }
    if out_path is not None:
        atomic_json(out_path, result)
    return result


def _collect_phase2_runs(
    results_dir: Optional[str | Path],
    backbone_filter: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, int, str], pd.DataFrame], List[Dict[str, Any]]]:
    manifest_rows: List[Dict[str, Any]] = []
    preds_map: Dict[Tuple[str, str, int, int, str], pd.DataFrame] = {}
    summary_source_rows: List[Dict[str, Any]] = []

    if results_dir is None:
        return manifest_rows, preds_map, summary_source_rows
    root = Path(results_dir)
    if not (root / "runs").is_dir():
        return manifest_rows, preds_map, summary_source_rows

    for meta_file in sorted((root / "runs").glob("*/metadata.json")):
        rdir = meta_file.parent
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("backbone") != backbone_filter or meta.get("run_mode") != "main":
            continue
        metrics = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        cost = json.loads((rdir / "cost.json").read_text(encoding="utf-8"))
        comp_path = rdir / "completion.json"
        comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.is_file() else {}
        pred_path = rdir / "predictions_test.csv"
        if not pred_path.is_file():
            continue

        direction = meta["direction"]
        method = meta["method"]
        fraction = int(meta["fraction"])
        seed = int(meta["seed"])
        preds_map[(backbone_filter, direction, fraction, seed, method)] = pd.read_csv(pred_path)

        primary_view = "V3-NEU/none" if direction == "SA" else "V7-VSMEC/none"
        test_m = (metrics.get("test_metrics") or {}).get(primary_view) or {}
        manifest_rows.append(
            {
                "backbone": backbone_filter,
                "direction": direction,
                "method": method,
                "fraction": fraction,
                "seed": seed,
                "lr": float(meta["lr"]),
                "run_id": meta["run_id"],
                "dev_selection_score": float(metrics["dev_selection_score"]),
                "test_primary_view": primary_view.split("/")[0],
                "test_primary_macro_f1": float(test_m.get("macro_f1", np.nan)),
                "test_primary_accuracy": float(test_m.get("accuracy", np.nan)),
                "best_epoch": metrics.get("best_epoch"),
                "best_step": int(metrics.get("best_step", 0)),
                "early_stopped": bool(metrics.get("early_stopped", False)),
                "actual_update_steps": int(cost.get("actual_update_steps", 0)),
                "train_seconds": float(cost.get("train_seconds", 0.0)),
                "gpu_hours": float(cost.get("gpu_hours", 0.0)),
                "peak_vram_mb": float(cost.get("peak_vram_mb", 0.0)),
                "total_params": int(cost.get("total_params", 0)),
                "trainable_params": int(cost.get("trainable_params", 0)),
                "gpu_name": cost.get("gpu_name", ""),
                "effective_precision": cost.get("effective_precision", ""),
                "ckpt_saved": bool(comp.get("ckpt_path")),
            }
        )
        for view_key, vm in (metrics.get("test_metrics") or {}).items():
            view_name, map_method = view_key.split("/", 1)
            summary_source_rows.append(
                {
                    "backbone": backbone_filter,
                    "direction": direction,
                    "method": method,
                    "fraction": fraction,
                    "seed": seed,
                    "label_view": view_name,
                    "map_method": map_method,
                    "n_eval": int(vm["n_eval"]),
                    "dev_selection_score": float(metrics["dev_selection_score"]),
                    "macro_f1": float(vm["macro_f1"]),
                    "weighted_f1": float(vm["weighted_f1"]),
                    "accuracy": float(vm["accuracy"]),
                    "train_seconds": float(cost.get("train_seconds", 0.0)),
                    "gpu_hours": float(cost.get("gpu_hours", 0.0)),
                    "peak_vram_mb": float(cost.get("peak_vram_mb", 0.0)),
                    "total_params": int(cost.get("total_params", 0)),
                    "trainable_params": int(cost.get("trainable_params", 0)),
                }
            )
    return manifest_rows, preds_map, summary_source_rows


def build_task11_tables_and_report(
    *,
    config_path: str | Path = "configs/phase2.yaml",
    xlmr_results_dir: Optional[str | Path] = None,
    xlmr_zs_dir: Optional[str | Path] = None,
    llm_results_dir: Optional[str | Path] = None,
    visobert_results_dir: Optional[str | Path] = None,
    encoder_cost_path: Optional[str | Path] = None,
    llm_cost_dir: Optional[str | Path] = None,
    out_dir: str | Path = "results/phase2/tables",
    report_path: str | Path = "reports/phase2/task11_xlmr_llm_10pct.md",
) -> Dict[str, Any]:
    """Synthesize Task 11 XLMR, LLM, and ViSoBERT 10% tables, exploratory contrasts, and cost report."""
    cfg = load_phase2_config(config_path)
    tables_dir = Path(out_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_out = Path(report_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    xlmr_manifest, xlmr_preds, xlmr_rows = _collect_phase2_runs(xlmr_results_dir, "XLMR")
    _, vis_preds, vis_rows = _collect_phase2_runs(visobert_results_dir, "ViSoBERT")

    # Collect XLMR ZS
    zs_dir_resolved = Path(xlmr_zs_dir) if xlmr_zs_dir is not None else (
        Path(xlmr_results_dir) / "xlmr_zs" if xlmr_results_dir is not None else None
    )
    if zs_dir_resolved is not None and (zs_dir_resolved / "runs").is_dir():
        for meta_file in sorted((zs_dir_resolved / "runs").glob("*/metadata.json")):
            rdir = meta_file.parent
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            metrics = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
            cost = json.loads((rdir / "cost.json").read_text(encoding="utf-8"))
            pred_path = rdir / "predictions_test.csv"
            if not pred_path.is_file():
                continue
            direction = meta["direction"]
            seed = int(meta["seed"])
            xlmr_preds[("XLMR", direction, 100, seed, "zs")] = pd.read_csv(pred_path)
            xlmr_manifest.append(
                {
                    "backbone": "XLMR",
                    "direction": direction,
                    "method": "zs",
                    "fraction": 100,
                    "seed": seed,
                    "lr": np.nan,
                    "run_id": meta["run_id"],
                    "dev_selection_score": np.nan,
                    "test_primary_view": "V3-NEU" if direction == "SA" else "V3-VSMEC",
                    "test_primary_macro_f1": float(
                        next(iter((metrics.get("test_metrics") or {}).values()), {}).get("macro_f1", np.nan)
                    ),
                    "test_primary_accuracy": float(
                        next(iter((metrics.get("test_metrics") or {}).values()), {}).get("accuracy", np.nan)
                    ),
                    "best_epoch": np.nan,
                    "best_step": 0,
                    "early_stopped": False,
                    "actual_update_steps": 0,
                    "train_seconds": 0.0,
                    "gpu_hours": 0.0,
                    "peak_vram_mb": np.nan,
                    "total_params": np.nan,
                    "trainable_params": 0,
                    "gpu_name": "",
                    "effective_precision": cost.get("precision", "fp32"),
                    "ckpt_saved": False,
                }
            )
            for view_key, vm in (metrics.get("test_metrics") or {}).items():
                view_name, map_method = view_key.split("/", 1)
                xlmr_rows.append(
                    {
                        "backbone": "XLMR",
                        "direction": direction,
                        "method": "zs",
                        "fraction": 100,
                        "seed": seed,
                        "label_view": view_name,
                        "map_method": map_method,
                        "n_eval": int(vm["n_eval"]),
                        "dev_selection_score": np.nan,
                        "macro_f1": float(vm["macro_f1"]),
                        "weighted_f1": float(vm["weighted_f1"]),
                        "accuracy": float(vm["accuracy"]),
                        "train_seconds": 0.0,
                        "gpu_hours": 0.0,
                        "peak_vram_mb": np.nan,
                        "total_params": np.nan,
                        "trainable_params": 0,
                    }
                )

    # Collect LLM test runs
    llm_rows: List[Dict[str, Any]] = []
    llm_preds: Dict[Tuple[str, str], pd.DataFrame] = {}
    llm_meta_summary: Dict[str, Any] = {}
    if llm_results_dir is not None:
        llm_root = Path(llm_results_dir)
        for domain, direction in (("neuesc", "SA"), ("vsmec", "SB")):
            dom_dir = llm_root / domain
            if not dom_dir.is_dir():
                continue
            for test_dir in sorted(dom_dir.glob("test_*")):
                if not test_dir.is_dir():
                    continue
                pred_file = test_dir / "predictions_test.csv"
                metrics_file = test_dir / "metrics.json"
                meta_file = test_dir / "metadata.json"
                if not (pred_file.is_file() and metrics_file.is_file() and meta_file.is_file()):
                    continue
                m_obj = json.loads(metrics_file.read_text(encoding="utf-8"))
                meta_obj = json.loads(meta_file.read_text(encoding="utf-8"))
                mode_key = str(meta_obj.get("mode", test_dir.name.replace("test_", "")))
                llm_preds[(direction, mode_key)] = pd.read_csv(pred_file)
                llm_meta_summary[f"{domain}/{mode_key}"] = {
                    "strategy": meta_obj.get("strategy"),
                    "prompt_style": meta_obj.get("prompt_style"),
                    "invalid_output_rate": m_obj.get("invalid_output_rate"),
                    "model_meta": meta_obj.get("model_meta", {}),
                }
                model_meta = meta_obj.get("model_meta") or {}
                for view_key, vm in (m_obj.get("test_metrics") or {}).items():
                    view_name, map_method = view_key.split("/", 1)
                    llm_rows.append(
                        {
                            "backbone": str(model_meta.get("model_name", "LLM")),
                            "direction": direction,
                            "method": f"llm-{mode_key}",
                            "fraction": 10,
                            "seed": 0,
                            "label_view": view_name,
                            "map_method": map_method,
                            "n_eval": int(vm["n_eval"]),
                            "dev_selection_score": np.nan,
                            "macro_f1": float(vm["macro_f1"]),
                            "weighted_f1": float(vm["weighted_f1"]),
                            "accuracy": float(vm["accuracy"]),
                            "train_seconds": 0.0,
                            "gpu_hours": 0.0,
                            "peak_vram_mb": np.nan,
                            "total_params": model_meta.get("total_params", np.nan),
                            "trainable_params": 0,
                        }
                    )

    manifest_cols = [
        "backbone", "direction", "method", "fraction", "seed", "lr", "run_id",
        "dev_selection_score", "test_primary_view", "test_primary_macro_f1", "test_primary_accuracy",
        "best_epoch", "best_step", "early_stopped", "actual_update_steps",
        "train_seconds", "gpu_hours", "peak_vram_mb", "total_params", "trainable_params",
        "gpu_name", "effective_precision", "ckpt_saved",
    ]
    manifest_df = pd.DataFrame(xlmr_manifest, columns=manifest_cols)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["direction", "fraction", "method", "seed"]).reset_index(drop=True)
    _atomic_csv(manifest_df, tables_dir / "task11_xlmr_manifest.csv")

    # 2. Metrics summary table
    all_eval_df = pd.DataFrame(vis_rows + xlmr_rows + llm_rows)
    summary_rows: List[Dict[str, Any]] = []
    if not all_eval_df.empty:
        grouped = all_eval_df.groupby(
            ["backbone", "direction", "method", "fraction", "label_view", "map_method"],
            dropna=False,
            sort=True,
        )
        for (bb, direction, method, fraction, view, mapping), grp in grouped:
            row: Dict[str, Any] = {
                "backbone": bb,
                "direction": direction,
                "method": method,
                "fraction": int(fraction),
                "n_seeds": int(grp["seed"].nunique()),
                "label_view": view,
                "map_method": mapping,
                "n_eval": int(grp["n_eval"].iloc[0]),
            }
            dev_vals = grp["dev_selection_score"].dropna().astype(float).to_numpy()
            row["dev_selection_mean"] = float(dev_vals.mean()) if len(dev_vals) else np.nan
            row["dev_selection_sd"] = float(dev_vals.std(ddof=1)) if len(dev_vals) > 1 else (0.0 if len(dev_vals) == 1 else np.nan)
            for stat_col in ("macro_f1", "weighted_f1", "accuracy"):
                vals = grp[stat_col].astype(float).to_numpy()
                row[f"{stat_col}_mean"] = float(vals.mean())
                row[f"{stat_col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    _atomic_csv(summary_df, tables_dir / "task11_metrics_summary.csv")

    # 3. Per-class F1 table (Strict VSMEC exports ONLY negative & positive, NEVER neutral)
    per_class_rows: List[Dict[str, Any]] = []
    all_preds_grouped: Dict[Tuple[str, str, int, str], List[Tuple[int, pd.DataFrame]]] = {}
    for (bb, direction, fraction, seed, method), df_pred in {**vis_preds, **xlmr_preds}.items():
        all_preds_grouped.setdefault((bb, direction, fraction, method), []).append((seed, df_pred))
    for (direction, mode_key), df_pred in llm_preds.items():
        all_preds_grouped.setdefault(("LLM", direction, 10, f"llm-{mode_key}"), []).append((0, df_pred))

    for (bb, direction, fraction, method), seed_frames in sorted(all_preds_grouped.items()):
        seed_frames.sort(key=lambda x: x[0])
        if method == "zs":
            view_list = [("V3-NEU", "sum"), ("V3-NEU", "argmax")] if direction == "SA" else [
                ("V3-VSMEC", "none"), ("V3-VSMEC-strict", "none")
            ]
        elif direction == "SA":
            view_list = [("V3-NEU", "none")]
        else:
            view_list = [("V7-VSMEC", "none"), ("V5-VSMEC", "none"), ("V3-VSMEC", "sum"), ("V3-VSMEC-strict", "sum")]

        for view, mapping in view_list:
            per_seed_f1 = []
            ref_labels: Optional[Tuple[int, ...]] = None
            for _, df_pred in seed_frames:
                _, y_true, y_pred, labels = _extract_view_arrays(df_pred, view, mapping)
                ref_labels = labels
                per_seed_f1.append(f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0))
            if not per_seed_f1 or ref_labels is None:
                continue
            mat = np.vstack(per_seed_f1)
            for idx_lbl, lbl_id in enumerate(ref_labels):
                lbl_name = EMOTIONS7[lbl_id] if view in ("V7-VSMEC", "V5-VSMEC") else ID_TO_LABEL3[lbl_id]
                vals = mat[:, idx_lbl]
                per_class_rows.append(
                    {
                        "backbone": bb,
                        "direction": direction,
                        "method": method,
                        "fraction": int(fraction),
                        "label_view": view,
                        "map_method": mapping,
                        "n_seeds": len(seed_frames),
                        "class_id": int(lbl_id),
                        "class_name": lbl_name,
                        "f1_mean": float(vals.mean()),
                        "f1_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    }
                )
    per_class_df = pd.DataFrame(per_class_rows)
    if not per_class_df.empty:
        strict_sub = per_class_df.loc[per_class_df["label_view"].eq("V3-VSMEC-strict")]
        if not strict_sub.empty:
            assert set(strict_sub["class_name"].tolist()) <= {"negative", "positive"}, (
                "Strict per-class F1 may ONLY export negative and positive!"
            )
    _atomic_csv(per_class_df, tables_dir / "task11_per_class_f1.csv")

    # 4. Exploratory paired bootstrap contrasts (NEVER altering Holm m=2 family)
    stats_cfg = cfg["statistics"]
    boot_n = int(stats_cfg["boot"])
    boot_seed = int(stats_cfg["boot_seed"])
    ci_level = float(stats_cfg["ci"])
    contrast_rows: List[Dict[str, Any]] = []

    def _build_arm(bb: str, direction: str, fraction: int, method: str, seeds_req: Sequence[int], view: str, mapping: str):
        preds_list = []
        ref_ids = ref_y = ref_labels = None
        for s in seeds_req:
            key = (bb, direction, fraction, s, method)
            if key not in xlmr_preds and key not in vis_preds:
                return None
            df_p = xlmr_preds.get(key) if key in xlmr_preds else vis_preds[key]
            ids, y_true, y_pred, labels = _extract_view_arrays(df_p, view, mapping)
            if ref_ids is None:
                ref_ids, ref_y, ref_labels = ids, y_true, labels
            else:
                assert ids.tolist() == ref_ids.tolist() and np.array_equal(ref_y, y_true)
            preds_list.append(y_pred)
        return {"ids": ref_ids, "y_true": ref_y, "preds": np.vstack(preds_list), "seeds": list(seeds_req), "labels": ref_labels}

    specs = [
        ("EXP_XLMR_SA_10_SEQ_MINUS_DFT", "XLMR", "SA", 10, "seq_minus_dft", "XLMR", "dft", 10, "XLMR", "seq", 10, [0, 1, 2], "V3-NEU", "none", "none"),
        ("EXP_XLMR_SB_10_SEQ_MINUS_DFT", "XLMR", "SB", 10, "seq_minus_dft", "XLMR", "dft", 10, "XLMR", "seq", 10, [0, 1, 2], "V7-VSMEC", "none", "none"),
        ("EXP_XLMR_SA_10_SEQ_MINUS_ZS_SUM", "XLMR", "SA", 10, "seq_minus_zs", "XLMR", "zs", 100, "XLMR", "seq", 10, [0, 1, 2], "V3-NEU", "sum", "none"),
        ("EXP_XLMR_SB_10_SEQ_MINUS_ZS_V3", "XLMR", "SB", 10, "seq_minus_zs", "XLMR", "zs", 100, "XLMR", "seq", 10, [0, 1, 2], "V3-VSMEC", "none", "sum"),
        ("EXP_XLMR_SB_10_SEQ_MINUS_ZS_STRICT", "XLMR", "SB", 10, "seq_minus_zs", "XLMR", "zs", 100, "XLMR", "seq", 10, [0, 1, 2], "V3-VSMEC-strict", "none", "sum"),
        ("EXP_SA_10_XLMR_MINUS_VISOBERT_DFT", "XLMR_vs_ViSoBERT", "SA", 10, "xlmr_minus_visobert_dft", "ViSoBERT", "dft", 10, "XLMR", "dft", 10, [0, 1, 2], "V3-NEU", "none", "none"),
        ("EXP_SA_10_XLMR_MINUS_VISOBERT_SEQ", "XLMR_vs_ViSoBERT", "SA", 10, "xlmr_minus_visobert_seq", "ViSoBERT", "seq", 10, "XLMR", "seq", 10, [0, 1, 2], "V3-NEU", "none", "none"),
        ("EXP_SB_10_XLMR_MINUS_VISOBERT_DFT", "XLMR_vs_ViSoBERT", "SB", 10, "xlmr_minus_visobert_dft", "ViSoBERT", "dft", 10, "XLMR", "dft", 10, [0, 1, 2], "V7-VSMEC", "none", "none"),
        ("EXP_SB_10_XLMR_MINUS_VISOBERT_SEQ", "XLMR_vs_ViSoBERT", "SB", 10, "xlmr_minus_visobert_seq", "ViSoBERT", "seq", 10, "XLMR", "seq", 10, [0, 1, 2], "V7-VSMEC", "none", "none"),
    ]
    for cid, bb_tag, direction, frac, cname, ref_bb, ref_m, ref_f, new_bb, new_m, new_f, seeds_req, view, ref_map, new_map in specs:
        ref_arm = _build_arm(ref_bb, direction, ref_f, ref_m, seeds_req, view, ref_map)
        new_arm = _build_arm(new_bb, direction, new_f, new_m, seeds_req, view, new_map)
        if ref_arm is None or new_arm is None:
            continue
        est = paired_gain(ref_arm, new_arm, ref_arm["labels"], seeds_req, boot=boot_n, seed=boot_seed, ci=ci_level, stat="macro_f1")
        contrast_rows.append(
            {
                "comparison_id": cid,
                "backbone": bb_tag,
                "direction": direction,
                "fraction": frac,
                "contrast": cname,
                "ref_method": f"{ref_bb}:{ref_m}",
                "new_method": f"{new_bb}:{new_m}",
                "label_view": view,
                "map_method": f"{ref_map}->{new_map}",
                "stat": "macro_f1",
                "n_seeds": len(seeds_req),
                "seeds": ",".join(map(str, seeds_req)),
                "ref_macro_f1": est["base_f1"],
                "new_macro_f1": est["seq_f1"],
                "delta_point": est["point"],
                "ci_lo": est["ci_lo"],
                "ci_hi": est["ci_hi"],
                "se": est["se"],
                "n_seed_pos": est["n_seed_pos"],
                "p_boot": est["p_boot"],
            }
        )
    contrast_rows = apply_phase2_holm(contrast_rows, cfg)
    contrasts_df = pd.DataFrame(contrast_rows)
    if not contrasts_df.empty:
        assert not contrasts_df["primary"].any() and contrasts_df["exploratory"].all()
        assert contrasts_df["p_holm"].isna().all()
    _atomic_csv(contrasts_df, tables_dir / "task11_exploratory_contrasts.csv")

    # 5. Cost vs performance table (10% reference comparison)
    enc_bench_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if encoder_cost_path is not None and Path(encoder_cost_path).is_file():
        enc_payload = json.loads(Path(encoder_cost_path).read_text(encoding="utf-8"))
        for b in enc_payload.get("benchmarks", []):
            enc_bench_map[(b["backbone"], b["domain"])] = b

    llm_cost_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    llm_c_dir = Path(llm_cost_dir) if llm_cost_dir is not None else (Path(llm_results_dir) if llm_results_dir is not None else None)
    if llm_c_dir is not None and llm_c_dir.is_dir():
        for domain in ("neuesc", "vsmec"):
            cfile = llm_c_dir / f"cost_llm_{domain}.json"
            if not cfile.is_file():
                continue
            cpayload = json.loads(cfile.read_text(encoding="utf-8"))
            modes_dict = cpayload.get("modes") or {"zero_shot": cpayload}
            for mode_key, mcost in modes_dict.items():
                llm_cost_map[(domain, f"llm-{mode_key}")] = {**cpayload, **mcost}

    cost_perf_rows: List[Dict[str, Any]] = []
    if not summary_df.empty:
        for direction, domain, prim_view, prim_map in (
            ("SA", "neuesc", "V3-NEU", "none"),
            ("SB", "vsmec", "V7-VSMEC", "none"),
        ):
            sub_s = summary_df.loc[
                summary_df["direction"].eq(direction)
                & summary_df["fraction"].eq(10)
                & summary_df["label_view"].eq(prim_view)
                & summary_df["map_method"].eq(prim_map)
            ]
            for _, r in sub_s.iterrows():
                bb = str(r["backbone"])
                method = str(r["method"])
                train_sub = [
                    x for x in (vis_rows + xlmr_rows)
                    if x["backbone"] == bb and x["direction"] == direction and x["method"] == method and int(x["fraction"]) == 10
                ]
                train_gpu_h = float(np.mean([x["gpu_hours"] for x in train_sub])) if train_sub else 0.0
                train_vram = float(np.max([x["peak_vram_mb"] for x in train_sub])) if train_sub else np.nan
                total_p = int(train_sub[0]["total_params"]) if train_sub else None
                train_p = int(train_sub[0]["trainable_params"]) if train_sub else 0

                eb = enc_bench_map.get((bb, domain))
                lc = llm_cost_map.get((domain, method))
                bench = eb or lc or {}
                if total_p is None and "total_params" in bench:
                    total_p = int(bench["total_params"])

                cost_perf_rows.append(
                    {
                        "backbone": bb,
                        "direction": direction,
                        "domain": domain,
                        "method": method,
                        "fraction": 10,
                        "primary_view": f"{prim_view}/{prim_map}",
                        "macro_f1_mean": float(r["macro_f1_mean"]),
                        "macro_f1_sd": float(r["macro_f1_sd"]),
                        "total_params": total_p,
                        "trainable_params": train_p,
                        "mean_train_gpu_hours": train_gpu_h,
                        "train_peak_vram_mb": train_vram,
                        "infer_precision": bench.get("precision", ""),
                        "infer_gpu_name": bench.get("gpu_name", ""),
                        "infer_peak_vram_mb": bench.get("peak_vram_mb", np.nan),
                        "latency_batch1_mean_ms": bench.get("latency_batch1_mean_ms", np.nan),
                        "latency_batch1_p95_ms": bench.get("latency_batch1_p95_ms", np.nan),
                        "large_batch_size": bench.get("large_batch_size", np.nan),
                        "throughput_samples_per_sec": bench.get("throughput_samples_per_sec", np.nan),
                        "infer_gpu_hours_per_1k": bench.get("gpu_hours_per_1000_samples", np.nan),
                    }
                )
    cost_perf_df = pd.DataFrame(cost_perf_rows)
    _atomic_csv(cost_perf_df, tables_dir / "task11_cost_performance_10pct.csv")

    # 6. Markdown Report
    lines = [
        "# Task 11 — Báo cáo Tham chiếu XLM-R-base & Open-Weight LLM (Mức 10%, `phase2-v1`)",
        "",
        f"- **Giao thức:** `{cfg['protocol_version']}`",
        "- **Quy tắc thống kê:** Mọi đối chiếu của `XLMR` và `LLM` trong báo cáo này đều là **thăm dò (`exploratory = True`)**; gia đình kiểm định xác nhận `Holm m = 2` thuộc riêng hai so sánh chính của `ViSoBERT` ở Task 08.",
        f"- **Cảnh báo rò rỉ dữ liệu tiền huấn luyện (LLM):** {PRETRAINING_LEAKAGE_NOTE}",
        "",
        "## 1. Tóm tắt hiệu năng trên tập `test` (góc nhìn chính theo miền)",
        "",
    ]
    if not summary_df.empty:
        lines.extend([
            "| Backbone | Chiều | Phương pháp | Mức | Seeds | View | `Macro-F1` | `Weighted-F1` | `Accuracy` |",
            "|---|---|---|---:|---:|---|---:|---:|---:|",
        ])
        for _, r in summary_df.iterrows():
            if (r["direction"] == "SA" and r["label_view"] == "V3-NEU") or (
                r["direction"] == "SB" and r["label_view"] in ("V7-VSMEC", "V3-VSMEC", "V3-VSMEC-strict")
            ):
                lines.append(
                    f"| `{r['backbone']}` | `{r['direction']}` | `{r['method']}` | `{int(r['fraction'])}%` | `{int(r['n_seeds'])}` | "
                    f"`{r['label_view']}/{r['map_method']}` | `{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}` | "
                    f"`{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_sd']:.4f}` | `{r['accuracy_mean']:.4f} ± {r['accuracy_sd']:.4f}` |"
                )
    else:
        lines.append("_Chưa có dữ liệu kết quả để tổng hợp._")

    lines.extend(["", "## 2. Đánh đổi Hiệu năng – Chi phí trên cùng GPU chuẩn", ""])
    if not cost_perf_df.empty:
        lines.extend([
            "| Backbone | Chiều | Phương pháp | `Macro-F1` | Total Params | Trainable | Train GPU-h | Infer Prec | Latency b=1 (ms) | Throughput (s/s) | GPU-h/1k |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ])
        for _, r in cost_perf_df.iterrows():
            lat = f"{r['latency_batch1_mean_ms']:.2f}" if pd.notna(r["latency_batch1_mean_ms"]) else "N/A"
            thr = f"{r['throughput_samples_per_sec']:.1f}" if pd.notna(r["throughput_samples_per_sec"]) else "N/A"
            gh1k = f"{r['infer_gpu_hours_per_1k']:.4f}" if pd.notna(r["infer_gpu_hours_per_1k"]) else "N/A"
            tot = f"{int(r['total_params']):,}" if pd.notna(r["total_params"]) else "N/A"
            lines.append(
                f"| `{r['backbone']}` | `{r['direction']}` | `{r['method']}` | `{r['macro_f1_mean']:.4f}` | "
                f"`{tot}` | `{int(r['trainable_params']):,}` | `{r['mean_train_gpu_hours']:.4f}` | "
                f"`{r['infer_precision'] or 'N/A'}` | `{lat}` | `{thr}` | `{gh1k}` |"
            )
    else:
        lines.append("_Chưa có dữ liệu chi phí suy luận._")

    lines.extend([
        "",
        "## 3. Ghi chú kiểm định & Quy tắc xuất liệu",
        "",
        "- `task11_per_class_f1.csv`: góc nhìn `V3-VSMEC-strict` chỉ xuất 2 lớp `negative` và `positive` theo đúng `AGENTS.md`.",
        "- `task11_exploratory_contrasts.csv`: dùng `paired_gain` (`boot=10000, boot_seed=0`), toàn bộ dòng gắn `primary=False, exploratory=True`.",
    ])
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "tables_dir": tables_dir,
        "report_path": report_out,
        "manifest_df": manifest_df,
        "summary_df": summary_df,
        "per_class_df": per_class_df,
        "contrasts_df": contrasts_df,
        "cost_perf_df": cost_perf_df,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_zs = sub.add_parser("xlmr-zs", help="Run XLMR zero-shot FP32 inference with completion/SKIP.")
    p_zs.add_argument("--config", default="configs/phase2.yaml")
    p_zs.add_argument("--results-dir", default="results/phase2")
    p_zs.add_argument("--ckpt-dir", default="checkpoints/phase2")
    p_zs.add_argument("--clean-dir", default="data/clean")
    p_zs.add_argument("--out-dir")
    p_zs.add_argument("--batch-size", type=int, default=32)
    p_zs.add_argument("--max-examples", type=int)

    p_be = sub.add_parser("benchmark-encoders", help="Benchmark ViSoBERT and XLMR inference on same GPU.")
    p_be.add_argument("--config", default="configs/phase2.yaml")
    p_be.add_argument("--clean-dir", default="data/clean")
    p_be.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p_be.add_argument("--n-b1", type=int, default=100)
    p_be.add_argument("--large-batch-size", type=int, default=32)
    p_be.add_argument("--n-large", type=int)
    p_be.add_argument("--out-path", default="results/phase2/cost_encoder_inference.json")

    p_sum = sub.add_parser("summarize", help="Generate Task 11 CSV tables and Markdown report.")
    p_sum.add_argument("--config", default="configs/phase2.yaml")
    p_sum.add_argument("--xlmr-results-dir")
    p_sum.add_argument("--xlmr-zs-dir")
    p_sum.add_argument("--llm-results-dir")
    p_sum.add_argument("--visobert-results-dir")
    p_sum.add_argument("--encoder-cost-path")
    p_sum.add_argument("--llm-cost-dir")
    p_sum.add_argument("--out-dir", default="results/phase2/tables")
    p_sum.add_argument("--report-path", default="reports/phase2/task11_xlmr_llm_10pct.md")

    args = parser.parse_args(argv)
    if args.action == "xlmr-zs":
        res = run_xlmr_zs_eval(
            args.results_dir,
            args.ckpt_dir,
            config_path=args.config,
            clean_dir=args.clean_dir,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
        )
        for rec in res["runs"]:
            print(f"{rec['run_id']}: {'SKIP' if rec['skipped'] else 'complete'}")
    elif args.action == "benchmark-encoders":
        res = benchmark_encoder_inference(
            config_path=args.config,
            clean_dir=args.clean_dir,
            precision=args.precision,
            n_b1=args.n_b1,
            large_batch_size=args.large_batch_size,
            n_large=args.n_large,
            out_path=args.out_path,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.action == "summarize":
        res = build_task11_tables_and_report(
            config_path=args.config,
            xlmr_results_dir=args.xlmr_results_dir,
            xlmr_zs_dir=args.xlmr_zs_dir,
            llm_results_dir=args.llm_results_dir,
            visobert_results_dir=args.visobert_results_dir,
            encoder_cost_path=args.encoder_cost_path,
            llm_cost_dir=args.llm_cost_dir,
            out_dir=args.out_dir,
            report_path=args.report_path,
        )
        print(f"Generated Task 11 tables in {res['tables_dir']} and report at {res['report_path']}")


if __name__ == "__main__":
    main()
