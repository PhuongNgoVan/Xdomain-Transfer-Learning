"""Build reproducible zero-shot analysis tables from saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.common.constants import EMOTIONS7, HARD_MERGE_GROUPS
from src.common.hashing import sha256_file
from src.common.schema import ZS_COLUMNS, ZS_PRED_COLUMNS
from src.stats.bootstrap import bootstrap_arm, bootstrap_contrast
from src.zs.analysis import (CLASS_NAMES, VIEW_LABELS, baseline_floors, confusion_counts,
                             contrast_stat, js_divergence, merge_labels7, row_normalize)
from src.zs.infer import evaluation_data_hash

ALIASES = ("PhoBERT", "ViSoBERT")
JOBS = {"zsA": ("VSMEC", "NEU"), "zsB": ("NEU", "VSMEC"),
        "refV": ("VSMEC", "VSMEC"), "refN": ("NEU", "NEU")}
EXPECTED = {"zsA": {("V3-NEU", "sum"), ("V3-NEU", "argmax")},
            "zsB": {("V3-VSMEC", "none"), ("V3-VSMEC-strict", "none")},
            "refV": {("V7-VSMEC", "none"), ("V3-VSMEC", "sum"), ("V3-VSMEC", "argmax"),
                     ("V3-VSMEC-strict", "sum"), ("V3-VSMEC-strict", "argmax")},
            "refN": {("V3-NEU", "none")}}
MAIN_COLUMNS = ("backbone", "direction", "target", "label_view", "n_eval", "n_seeds", "zs_f1", "zs_std", "zs_ci_lo", "zs_ci_hi", "ref_f1", "ref_std", "ref_ci_lo", "ref_ci_hi", "gap", "gap_ci_lo", "gap_ci_hi", "retention", "ret_ci_lo", "ret_ci_hi", "floor_majority", "floor_prior_random")
ASYM_COLUMNS = ("kind", "contrast", "label_view", "stat", "diff", "ci_lo", "ci_hi")
JS_COLUMNS = ("kind", "backbone", "direction", "label_view", "role", "js_pred_true_bits", "std", "ci_lo", "ci_hi")
CLASS_COLUMNS = ("backbone", "direction", "label_view", "role", "class", "f1", "std", "ci_lo", "ci_hi")
MAPPING_COLUMNS = ("backbone", "label_view", "role", "map_method", "f1", "std", "ci_lo", "ci_hi")
FLOOR_COLUMNS = ("label_view", "floor", "macro_f1", "n_sim")


def _job(row):
    for name, (source, target) in JOBS.items():
        if str(row.stage1) == source and str(row.eval_domain) == target:
            return name
    raise ValueError(f"Cannot identify job for run {row.run_id}")


def _read_inputs(results_dir: Path, allow_missing: bool):
    path = results_dir / "zs.csv"
    metadata = pd.read_csv(path, keep_default_na=False)
    if tuple(metadata.columns) != tuple(ZS_COLUMNS):
        raise ValueError("zs.csv does not conform to ZS_COLUMNS")
    if metadata.empty:
        raise ValueError("zs.csv is empty")
    if metadata.duplicated(["run_id", "label_view", "map_method"]).any():
        raise ValueError("Duplicate zero-shot result key")
    if not metadata["split"].eq("test").all() or not metadata["precision"].eq("fp32").all():
        raise ValueError("Analysis accepts only FP32 test results")
    seed_values = pd.to_numeric(metadata["seed"], errors="coerce")
    if not seed_values.isin(range(5)).all() or not metadata.backbone.isin(ALIASES).all():
        raise ValueError("Analysis requires canonical backbone aliases and seed IDs 0..4")
    metadata["job"] = [_job(row) for row in metadata.itertuples()]
    missing, complete = [], []
    for alias in ALIASES:
        for job in JOBS:
            seeds = []
            for seed in range(5):
                rows = metadata[(metadata.backbone == alias) & (metadata.job == job) & (metadata.seed.astype(int) == seed)]
                actual = set(zip(rows.label_view, rows.map_method))
                if actual == EXPECTED[job] and rows.run_id.nunique() == 1 and len(rows) == len(actual):
                    seeds.append(seed)
                    complete.extend(rows.index.tolist())
            absent = sorted(set(range(5)) - set(seeds))
            if absent:
                missing.append(f"{alias}/{job}: missing seeds {absent}")
    if missing:
        message = "; ".join(missing)
        if not allow_missing:
            raise ValueError(message)
        print("[ALLOW MISSING] " + message)
    metadata = metadata.loc[sorted(set(complete))].copy()
    if metadata.empty:
        raise ValueError("No complete backbone/seed/job runs available for analysis")
    predictions = {}
    for run_id, rows in metadata.groupby("run_id", sort=True):
        if rows[["backbone", "seed", "job", "data_hash"]].nunique().gt(1).any():
            raise ValueError(f"Inconsistent metadata for {run_id}")
        row = rows.iloc[0]
        eval_set = "neuesc_test" if row.eval_domain == "NEU" else "vsmec_test"
        pred = pd.read_csv(results_dir / "predictions_zs" / f"{run_id}_{eval_set}.csv")
        if tuple(pred.columns) != tuple(ZS_PRED_COLUMNS) or pred.empty:
            raise ValueError(f"Invalid prediction schema for {run_id}")
        if pred.id.isna().any() or pred.id.duplicated().any():
            raise ValueError(f"Missing or duplicate prediction ID for {run_id}")
        pred = pred.sort_values("id", kind="stable").reset_index(drop=True)
        if not pred.y_true3.isin([0, 1, 2]).all():
            raise ValueError(f"Invalid true labels for {run_id}")
        for column in ("y_pred3_sum", "y_pred3_argmax"):
            if not pred[column].isin([0, 1, 2]).all():
                raise ValueError(f"Invalid predicted labels for {run_id}")
        probs = pred[["p3_neg", "p3_neu", "p3_pos"]].to_numpy(dtype=float)
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Invalid probabilities for {run_id}")
        if not np.array_equal(probs.argmax(axis=1), pred.y_pred3_sum.to_numpy()):
            raise ValueError(f"Predictions differ from saved probability argmax for {run_id}")
        if row.eval_domain == "VSMEC":
            if not pred.y_true7.isin(range(7)).all() or not np.array_equal(merge_labels7(pred.y_true7), pred.y_true3):
                raise ValueError(f"Invalid frozen VSMEC label mapping for {run_id}")
            strict = pred.y_true7.le(4).to_numpy()
            flags = pred.in_strict.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
            if not np.array_equal(strict, flags):
                raise ValueError(f"Invalid strict mask for {run_id}")
        if row.stage1 == "VSMEC":
            p7 = pred[[f"p7_{i}" for i in range(7)]].to_numpy(dtype=float)
            if not np.isfinite(p7).all() or (p7 < 0).any() or not np.allclose(p7.sum(axis=1), 1, atol=1e-6):
                raise ValueError(f"Invalid emotion probabilities for {run_id}")
            merged = np.stack([p7[:, HARD_MERGE_GROUPS[label]].sum(axis=1) for label in range(3)], axis=1)
            if not np.allclose(merged, probs, atol=1e-6):
                raise ValueError(f"Saved p3 differs from summed p7 probabilities for {run_id}")
            if not np.array_equal(p7.argmax(axis=1), pred.y_pred7.to_numpy()):
                raise ValueError(f"Invalid emotion predictions for {run_id}")
            if not np.array_equal(merge_labels7(pred.y_pred7), pred.y_pred3_argmax):
                raise ValueError(f"Invalid argmax-then-map predictions for {run_id}")
        elif not pred.y_pred3_sum.equals(pred.y_pred3_argmax):
            raise ValueError(f"Three-class prediction methods must agree for {run_id}")
        for result in rows.itertuples():
            n_eval = int(pred.y_true7.le(4).sum()) if result.label_view.endswith("strict") else len(pred)
            if int(result.n_eval) != n_eval:
                raise ValueError(f"n_eval does not match predictions for {run_id}/{result.label_view}")
        predictions[run_id] = pred
    return metadata, predictions, missing


def _build_arm(rows, predictions, view, method):
    rows = rows[(rows.label_view == view) & (rows.map_method == method)].sort_values("seed")
    if rows.empty:
        return None
    ids, truth, preds, seed_ids = None, None, [], []
    for row in rows.itertuples():
        frame = predictions[row.run_id]
        if view.endswith("strict"):
            frame = frame[frame.y_true7.le(4)]
        column = "y_pred7" if view == "V7-VSMEC" else ("y_pred3_argmax" if method == "argmax" else "y_pred3_sum")
        y = frame.y_true7.to_numpy(dtype=int) if view == "V7-VSMEC" else frame.y_true3.to_numpy(dtype=int)
        if ids is None:
            ids, truth = frame.id.to_numpy(), y
        elif not np.array_equal(ids, frame.id.to_numpy()) or not np.array_equal(truth, y):
            raise ValueError("Predictions across seeds must have identical test IDs and labels")
        preds.append(frame[column].to_numpy(dtype=int))
        seed_ids.append(int(row.seed))
    if len(set(seed_ids)) != len(seed_ids) or not len(truth):
        raise ValueError("Duplicate seed or empty label view")
    return {"y_true": truth, "preds_by_seed": np.stack(preds), "labels": VIEW_LABELS[view],
            "stat": "macro_f1", "ids": ids, "seeds": seed_ids}


def _assert_same_test(arms):
    first = next(iter(arms.values()))
    for arm in arms.values():
        if not np.array_equal(first["ids"], arm["ids"]) or not np.array_equal(first["y_true"], arm["y_true"]):
            raise ValueError("Contrast arms must share identical test IDs and labels")


def _arm_spec(arm, stat="macro_f1"):
    return {key: arm[key] for key in ("y_true", "preds_by_seed", "labels")} | {"stat": stat}


def _write_matrix(out_dir, name, counts, rows, columns):
    for suffix, values in (("counts", counts), ("rownorm", row_normalize(counts))):
        frame = pd.DataFrame(values, index=rows, columns=columns)
        frame.index.name = "label"
        frame.to_csv(out_dir / f"{name}_{suffix}.csv", lineterminator="\n")


def analyze_results(results_dir="results", data_dir="data/clean", out_dir=None, *, boot=10000,
                    seed=0, ci=0.95, allow_missing=False, n_sim=2000):
    """Recompute every point estimate from predictions and attach two-level CIs."""
    results_dir, data_dir = Path(results_dir), Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else results_dir / "tables"
    metadata, predictions, missing = _read_inputs(results_dir, allow_missing)
    evaluation_hashes = {}
    for target in metadata.eval_domain.unique():
        expected_hash = evaluation_data_hash(data_dir, "neuesc" if target == "NEU" else "vsmec")
        evaluation_hashes[target] = expected_hash
        target_rows = metadata[metadata.eval_domain == target]
        if not target_rows.data_hash.eq(expected_hash).all():
            stale_runs = sorted(target_rows.loc[target_rows.data_hash != expected_hash, "run_id"].unique())
            raise ValueError(f"Saved data_hash does not match current {target} evaluation data: {stale_runs}")
    # Floors consume the current VSMEC train labels, so verify those bytes too.
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(data_dir / "vsmec_train.csv") != manifest.get("vsmec_train.csv"):
        raise ValueError("VSMEC floor training data_hash does not match frozen MANIFEST.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_kwargs = {"boot": int(boot), "boot_seed": int(seed), "ci": float(ci)}
    if boot <= 0 or not 0 < ci < 1:
        raise ValueError("boot must be positive and ci must lie in (0,1)")
    train_neu = pd.read_csv(data_dir / "neuesc_100.csv", usecols=["subset", "label"])
    train_neu = train_neu.loc[train_neu.subset == "train", "label"].to_numpy(dtype=int)
    train_vsmec = merge_labels7(pd.read_csv(data_dir / "vsmec_train.csv", usecols=["label"]).label)
    priors = {"NEU": np.bincount(train_neu, minlength=3), "VSMEC": np.bincount(train_vsmec, minlength=3)}
    main_rows, js_rows, class_rows, mapping_rows, floors_rows = [], [], [], [], []
    comparisons, summary_cache, floor_cache, canonical_views = {}, {}, {}, {}
    contrasts_nan, bootstrap_se = {}, {}
    def summary(arm, stat="macro_f1"):
        digest = hashlib.sha256(arm["y_true"].tobytes() + arm["preds_by_seed"].tobytes()).hexdigest()
        cache_key = (digest, tuple(arm["labels"]), stat)
        if cache_key not in summary_cache:
            summary_cache[cache_key] = bootstrap_arm(arm["y_true"], arm["preds_by_seed"], arm["labels"], stat=stat, **bootstrap_kwargs)
        return summary_cache[cache_key]
    for alias in ALIASES:
        subset = metadata[metadata.backbone == alias]
        for direction, view, jobs in (("A", "V3-NEU", ("zsA", "refN")),
                                      ("B", "V3-VSMEC", ("zsB", "refV")),
                                      ("B", "V3-VSMEC-strict", ("zsB", "refV"))):
            arms = {}
            for role, job in zip(("zs", "ref"), jobs):
                method = "sum" if job in ("zsA", "refV") else "none"
                arm = _build_arm(subset[subset.job == job], predictions, view, method)
                if arm is None:
                    continue
                if view in canonical_views:
                    _assert_same_test({"first": canonical_views[view], "current": arm})
                else:
                    canonical_views[view] = arm
                arms[role] = arm
                result = summary(arm)
                js_result = summary(arm, "js")
                summary_key = f"{alias}/{direction}/{view}/{role}"
                bootstrap_se[summary_key + "/macro_f1"] = result["se"]
                bootstrap_se[summary_key + "/js"] = js_result["se"]
                js_rows.append({"kind": "prediction", "backbone": alias, "direction": direction,
                                "label_view": view, "role": role, "js_pred_true_bits": js_result["point"],
                                "std": js_result["std"], "ci_lo": js_result["ci_lo"], "ci_hi": js_result["ci_hi"]})
                for label in VIEW_LABELS[view]:
                    class_result = summary(arm, f"f1_{label}")
                    bootstrap_se[summary_key + f"/f1_{label}"] = class_result["se"]
                    class_rows.append({"backbone": alias, "direction": direction, "label_view": view, "role": role,
                                       "class": CLASS_NAMES[label], "f1": class_result["point"], "std": class_result["std"],
                                       "ci_lo": class_result["ci_lo"], "ci_hi": class_result["ci_hi"]})
                counts = confusion_counts(arm["y_true"], arm["preds_by_seed"], 3, 3)
                _write_matrix(out_dir, f"zs_confusion3_{alias}_{direction}_{view}_{role}", counts,
                              ["neg", "neu", "pos"], ["neg", "neu", "pos"])
                if view not in floor_cache:
                    train = train_neu if direction == "A" else train_vsmec
                    floor_cache[view] = baseline_floors(arm["y_true"], train, VIEW_LABELS[view], n_sim=n_sim, seed=seed)
                if job in ("zsA", "refV"):
                    for alternative in ("sum", "argmax"):
                        mapped_arm = arm if alternative == method else _build_arm(subset[subset.job == job], predictions, view, alternative)
                        mapped = summary(mapped_arm)
                        bootstrap_se[summary_key + f"/mapping_{alternative}"] = mapped["se"]
                        mapping_rows.append({"backbone": alias, "label_view": view, "role": role, "map_method": alternative,
                                             "f1": mapped["point"], "std": mapped["std"], "ci_lo": mapped["ci_lo"], "ci_hi": mapped["ci_hi"]})
            if set(arms) == {"zs", "ref"}:
                _assert_same_test(arms)
                comparisons[(alias, direction, view)] = arms
                estimates = {role: summary(arm) for role, arm in arms.items()}
                contrast_results = {}
                for stat in ("gap", "retention"):
                    contrast_results[stat] = bootstrap_contrast({role: _arm_spec(arm) for role, arm in arms.items()},
                        lambda x, stat=stat: contrast_stat(x, stat), share_test=True, **bootstrap_kwargs)
                    contrasts_nan[f"{alias}/{direction}/{view}/{stat}"] = contrast_results[stat]["n_nan"]
                    bootstrap_se[f"{alias}/{direction}/{view}/{stat}"] = contrast_results[stat]["se"]
                zs, ref = estimates["zs"], estimates["ref"]
                gap, ret = contrast_results["gap"], contrast_results["retention"]
                floors = floor_cache[view]
                main_rows.append({"backbone": alias, "direction": direction, "target": "NEU" if direction == "A" else "VSMEC",
                                  "label_view": view, "n_eval": len(arms["zs"]["y_true"]),
                                  "n_seeds": min(len(a["seeds"]) for a in arms.values()),
                                  "zs_f1": zs["point"], "zs_std": zs["std"], "zs_ci_lo": zs["ci_lo"], "zs_ci_hi": zs["ci_hi"],
                                  "ref_f1": ref["point"], "ref_std": ref["std"], "ref_ci_lo": ref["ci_lo"], "ref_ci_hi": ref["ci_hi"],
                                  "gap": gap["point"], "gap_ci_lo": gap["ci_lo"], "gap_ci_hi": gap["ci_hi"],
                                  "retention": ret["point"], "ret_ci_lo": ret["ci_lo"], "ret_ci_hi": ret["ci_hi"],
                                  "floor_majority": floors["majority"], "floor_prior_random": floors["prior_random"]})
        for job, n_true, n_pred, true_column, pred_column, matrix_tag in (
                ("zsB", 7, 3, "y_true7", "y_pred3_sum", "7x3"),
                ("zsA", 3, 7, "y_true3", "y_pred7", "3x7")):
            runs = subset[subset.job == job].drop_duplicates("run_id").sort_values("seed")
            if len(runs):
                frames = [predictions[r] for r in runs.run_id]
                ids, truth = frames[0].id.to_numpy(), frames[0][true_column].to_numpy(dtype=int)
                for frame in frames[1:]:
                    if not np.array_equal(ids, frame.id) or not np.array_equal(truth, frame[true_column]):
                        raise ValueError("Cross-label matrix inputs have inconsistent test IDs")
                counts = confusion_counts(truth, np.stack([f[pred_column].to_numpy(dtype=int) for f in frames]), n_true, n_pred)
                names_true = list(EMOTIONS7) if n_true == 7 else ["neg", "neu", "pos"]
                names_pred = list(EMOTIONS7) if n_pred == 7 else ["neg", "neu", "pos"]
                _write_matrix(out_dir, f"zs_matrix_{matrix_tag}_{alias}", counts, names_true, names_pred)
    for view, floors in sorted(floor_cache.items()):
        for floor in ("majority", "prior_random"):
            floors_rows.append({"label_view": view, "floor": floor, "macro_f1": floors[floor],
                                "n_sim": 1 if floor == "majority" else floors["n_sim"]})
    js_rows.append({"kind": "prior", "backbone": "", "direction": "", "label_view": "V3-VSMEC vs V3-NEU", "role": "",
                    "js_pred_true_bits": js_divergence(priors["VSMEC"], priors["NEU"]), "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    asymmetry_rows = []
    def asymmetry(kind, contrast, view, left_key, right_key):
        if left_key not in comparisons or right_key not in comparisons:
            return
        left, right = comparisons[left_key], comparisons[right_key]
        if left_key[1] == right_key[1]:
            _assert_same_test({**{f"l{k}": a for k, a in left.items()}, **{f"r{k}": a for k, a in right.items()}})
        arms = {**{f"l{k}": _arm_spec(a) for k, a in left.items()}, **{f"r{k}": _arm_spec(a) for k, a in right.items()}}
        share = {name: ("same" if left_key[1] == right_key[1] else name[0]) for name in arms}
        for stat in ("retention", "gap"):
            def difference(x, stat=stat):
                return contrast_stat({"zs": x["lzs"], "ref": x["lref"]}, stat) - contrast_stat({"zs": x["rzs"], "ref": x["rref"]}, stat)
            result = bootstrap_contrast(arms, difference, share_test=share, **bootstrap_kwargs)
            asymmetry_rows.append({"kind": kind, "contrast": contrast, "label_view": view, "stat": stat,
                                   "diff": result["point"], "ci_lo": result["ci_lo"], "ci_hi": result["ci_hi"]})
            contrasts_nan[f"asymmetry/{kind}/{contrast}/{view}/{stat}"] = result["n_nan"]
            bootstrap_se[f"asymmetry/{kind}/{contrast}/{view}/{stat}"] = result["se"]
    for alias in ALIASES:
        for view in ("V3-VSMEC", "V3-VSMEC-strict"):
            asymmetry("direction", f"{alias}:A-B", view, (alias, "A", "V3-NEU"), (alias, "B", view))
    for direction, view in (("A", "V3-NEU"), ("B", "V3-VSMEC"), ("B", "V3-VSMEC-strict")):
        asymmetry("backbone", f"{direction}:PhoBERT-ViSoBERT", view, ("PhoBERT", direction, view), ("ViSoBERT", direction, view))
    tables = {"zs_main.csv": (main_rows, MAIN_COLUMNS), "zs_asymmetry.csv": (asymmetry_rows, ASYM_COLUMNS),
              "zs_js.csv": (js_rows, JS_COLUMNS), "zs_per_class.csv": (class_rows, CLASS_COLUMNS),
              "zs_mapping_sensitivity.csv": (mapping_rows, MAPPING_COLUMNS), "zs_floors.csv": (floors_rows, FLOOR_COLUMNS)}
    frames = {}
    for name, (rows, columns) in tables.items():
        frame = pd.DataFrame(rows, columns=columns)
        frame.to_csv(out_dir / name, index=False, lineterminator="\n")
        frames[name] = frame
    try:
        code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_version = "unknown"
    data_files = ("vsmec_train.csv", "vsmec_test.csv", "neuesc_100.csv")
    hashes = {name: sha256_file(data_dir / name) for name in data_files}
    meta = {"boot": int(boot), "boot_seed": int(seed), "ci": float(ci), "code_version": code_version,
            "data_hash": hashes, "prediction_data_hashes": sorted(metadata.data_hash.unique().tolist()),
            "evaluation_data_hashes": evaluation_hashes,
            "n_rows": len(metadata), "metric_definitions_version": "zs-v3.0-1", "missing_groups": missing,
            "n_nan": contrasts_nan, "bootstrap_se": bootstrap_se,
            "n_seeds_by_arm": {"/".join(key) + "/" + role: len(a["seeds"]) for key, arms in comparisons.items() for role, a in arms.items()}}
    (out_dir / "zs_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"=== [ANALYZE DONE] rows={len(metadata)}, comparisons={len(main_rows)}, out={out_dir} ===")
    return {"tables": frames, "meta": meta, "out_dir": out_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze saved zero-shot predictions with two-level bootstrap CIs")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    analyze_results(args.results_dir, args.data_dir, args.out_dir, boot=args.boot, seed=args.seed, allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
