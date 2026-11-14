"""Recompute sequential gains from aligned saved predictions, with fixed Holm family."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from src.common.constants import EMOTIONS7
from src.common.hashing import sha256_file
from src.common.schema import SEQ_COLUMNS, ZS_PRED_COLUMNS
from src.seq.analysis import apply_primary_holm, paired_gain
from src.seq.protocol import atomic_json, digest, load_config
from src.zs.analysis import confusion_counts, merge_labels7, row_normalize
from src.zs.analyze import _read_inputs
from src.zs.infer import _git_commit
from src.zs.views import VIEW_LABELS

MAIN_COLUMNS = ("backbone", "direction", "label_view", "stat", "n_eval", "base_f1", "base_std", "seq_f1", "seq_std", "gain", "gain_ci_lo", "gain_ci_hi", "n_seed_pos", "p_boot", "p_holm", "significant", "primary", "exploratory", "n_seeds")
CLASS_COLUMNS = ("backbone", "direction", "label_view", "class", "base_f1", "seq_f1", "diff", "ci_lo", "ci_hi", "exploratory")
C0_COLUMNS = ("backbone", "direction", "label_view", "base_f1", "c0_f1", "seq_f1", "gain_c0", "gain_c0_ci_lo", "gain_c0_ci_hi", "s_minus_c0", "s_minus_c0_ci_lo", "s_minus_c0_ci_hi", "n_seeds", "exploratory")
VERSUS_COLUMNS = ("backbone", "direction", "label_view", "f1_base", "f1_zs", "f1_seq", "gain", "gap_step2")


def _arm(rows, predictions, view, method, seed_ids):
    picked = rows.loc[rows.label_view.eq(view) & rows.map_method.eq(method)]
    frames = []
    for s in seed_ids:
        one = picked.loc[picked.seed.eq(s)]
        if len(one) != 1:
            raise ValueError(f"Need exactly one {view}/{method}/seed{s} result")
        frame = predictions[one.iloc[0].run_id].sort_values("id").reset_index(drop=True)
        if view.endswith("strict"):
            frame = frame.loc[frame.y_true7.le(4)]
        if frame.empty:
            raise ValueError("Empty comparison view")
        frames.append(frame)
    original = frames[0]
    truth_col = "y_true7" if view == "V7-VSMEC" else "y_true3"
    pred_col = "y_pred7" if view == "V7-VSMEC" else ("y_pred3_argmax" if method == "argmax" else "y_pred3_sum")
    if any(not original.id.equals(f.id) or not original[truth_col].equals(f[truth_col]) for f in frames[1:]):
        raise ValueError("Inconsistent test IDs/labels across seeds")
    return {"ids": original.id.to_numpy(), "y_true": original[truth_col].to_numpy(dtype=int),
            "preds": np.stack([f[pred_col].to_numpy(dtype=int) for f in frames]), "seeds": list(seed_ids)}


def _slice(arm, seeds):
    return {**arm, "preds": arm["preds"][[arm["seeds"].index(s) for s in seeds]], "seeds": list(seeds)}


def analyze_results(results_dir, zs_results_dir, data_dir="data/clean", out_dir=None, *, config_path="configs/seq.yaml", boot=None, seed=None, allow_missing=False):
    cfg = load_config(config_path)
    if not cfg["baseline_provenance"].get("actual_run_verified", False):
        print("WARN: baseline run protocol unverified; numerical contrasts are not proof of protocol equivalence")
    root, zsroot = Path(results_dir), Path(zs_results_dir)
    output = Path(out_dir) if out_dir is not None else root / "tables_seq"
    boot = int(boot if boot is not None else cfg["boot"])
    seed = int(seed if seed is not None else cfg["boot_seed"])
    kwargs = {"boot": boot, "seed": seed, "ci": cfg["ci"]}
    metadata = pd.read_csv(root / "seq.csv", keep_default_na=False)
    if tuple(metadata.columns) != SEQ_COLUMNS or metadata.duplicated(["run_id", "split", "label_view", "map_method"]).any():
        raise ValueError("Invalid seq schema or duplicate result keys")
    metadata = metadata.loc[metadata.split.eq("test") & metadata.variant.str.startswith("seq-")].copy()
    selection = json.loads((root / "selection_seq.json").read_text(encoding="utf-8"))
    if selection["protocol_hash"] != digest(cfg["training"]):
        raise ValueError("Selection protocol mismatch")
    baselines, base_predictions, _ = _read_inputs(zsroot, allow_missing=True)
    predictions, missing = {}, []
    for run_id, group in metadata.groupby("run_id"):
        row = group.iloc[0]
        if group[["seed", "backbone", "direction", "data_hash", "lr"]].nunique().gt(1).any():
            raise ValueError("Inconsistent run metadata")
        mapped_direction = {"C0N": "SA", "C0V": "SB"}.get(row.direction, row.direction)
        if float(row.lr) != selection["selected"][f"{row.backbone}/{mapped_direction}"]["lr"]:
            raise ValueError("Results do not use the selected LR")
        sidecar = json.loads((root / "runs_seq" / f"{run_id}.json").read_text(encoding="utf-8"))
        if not sidecar["complete"] or sidecar["protocol_hash"] != digest(cfg["training"]):
            raise ValueError("Incomplete or incompatible training protocol")
        if digest(sidecar["data_hashes"]) != row.data_hash or any(sha256_file(Path(data_dir) / f) != h for f, h in sidecar["data_hashes"].items()):
            raise ValueError("Sequential data hash differs from frozen data")
        if any(not Path(p).is_file() or sha256_file(p) != h for p, h in sidecar["artifacts"].items()):
            raise ValueError("Run artifacts missing/changed")
        target = "neuesc" if mapped_direction == "SA" else "vsmec"
        frame = pd.read_csv(root / "predictions_seq" / f"{run_id}_{target}_test.csv")
        if tuple(frame.columns) != ZS_PRED_COLUMNS or frame.id.duplicated().any() or frame.id.isna().any():
            raise ValueError("Invalid sequential prediction schema/IDs")
        if not frame.y_true3.isin(range(3)).all() or not frame.y_pred3_sum.isin(range(3)).all():
            raise ValueError("Invalid polarity labels")
        if target == "vsmec" and (not frame.y_pred7.isin(range(7)).all() or not np.array_equal(merge_labels7(frame.y_true7), frame.y_true3)):
            raise ValueError("Invalid seven-class labels/mapping")
        predictions[run_id] = frame
    main_rows, classes, controls, comparisons = [], [], [], {}
    matrices = []
    for alias in cfg["backbones"]:
        for direction, main_view, ref_job in (("SA", "V3-NEU", "refN"), ("SB", "V7-VSMEC", "refV")):
            seq_rows = metadata.loc[metadata.backbone.eq(alias) & metadata.direction.eq(direction)]
            ref_rows = baselines.loc[baselines.backbone.eq(alias) & baselines.job.eq(ref_job)]
            seeds = sorted(set(seq_rows.seed.astype(int)) & set(ref_rows.seed.astype(int)))
            absent = sorted(set(cfg["seeds"]) - set(seeds))
            if absent:
                missing.append(f"{alias}/{direction}: missing paired seeds {absent}")
                if not allow_missing:
                    raise ValueError(missing[-1])
                print("WARN: " + missing[-1])
            if not seeds:
                continue
            views = [(main_view, "none")] if direction == "SA" else [(main_view, "none"), ("V3-VSMEC", "sum"), ("V3-VSMEC-strict", "sum")]
            for view, method in views:
                base = _arm(ref_rows, base_predictions, view, method, seeds)
                seq = _arm(seq_rows, predictions, view, method, seeds)
                labels = VIEW_LABELS[view]
                for stat in ("macro_f1", "weighted_f1", "accuracy"):
                    gain = paired_gain(base, seq, labels, seeds, stat=stat, **kwargs)
                    main_rows.append(dict(backbone=alias, direction=direction, label_view=view, stat=stat,
                                          n_eval=len(seq["ids"]), base_f1=gain["base_f1"], base_std=gain["base_std"], seq_f1=gain["seq_f1"], seq_std=gain["seq_std"],
                                          gain=gain["point"], gain_ci_lo=gain["ci_lo"], gain_ci_hi=gain["ci_hi"], n_seed_pos=gain["n_seed_pos"], p_boot=gain["p_boot"], n_seeds=len(seeds)))
                comparisons[(alias, direction, view)] = main_rows[-3]
                for label in labels:
                    gain = paired_gain(base, seq, labels, seeds, stat=f"f1_{label}", **kwargs)
                    classes.append(dict(backbone=alias, direction=direction, label_view=view, **{"class": EMOTIONS7[label] if view == "V7-VSMEC" else ("neg", "neu", "pos")[label]},
                                        base_f1=gain["base_f1"], seq_f1=gain["seq_f1"], diff=gain["point"], ci_lo=gain["ci_lo"], ci_hi=gain["ci_hi"], exploratory=True))
                c0_name = "C0N" if direction == "SA" else "C0V"
                c0_rows = metadata.loc[metadata.backbone.eq(alias) & metadata.direction.eq(c0_name)]
                common = sorted(set(seeds) & set(c0_rows.seed.astype(int)))
                if common:
                    c0 = _arm(c0_rows, predictions, view, method, common)
                    cb, cs = _slice(base, common), _slice(seq, common)
                    gc = paired_gain(cb, c0, labels, common, **kwargs)
                    sc = paired_gain(c0, cs, labels, common, **kwargs)
                    controls.append(dict(backbone=alias, direction=direction, label_view=view, base_f1=gc["base_f1"], c0_f1=gc["seq_f1"], seq_f1=sc["seq_f1"],
                                         gain_c0=gc["point"], gain_c0_ci_lo=gc["ci_lo"], gain_c0_ci_hi=gc["ci_hi"], s_minus_c0=sc["point"], s_minus_c0_ci_lo=sc["ci_lo"], s_minus_c0_ci_hi=sc["ci_hi"], n_seeds=len(common), exploratory=True))
                if view == main_view:
                    dimension = 3 if direction == "SA" else 7
                    matrices.append((alias, direction, f"{dimension}x{dimension}", confusion_counts(seq["y_true"], seq["preds"], dimension, dimension), list(EMOTIONS7) if dimension == 7 else ["neg", "neu", "pos"]))
                    if dimension == 7:
                        matrices.append((alias, direction, "7x3", confusion_counts(seq["y_true"], merge_labels7(seq["preds"]), 7, 3), list(EMOTIONS7)))
    apply_primary_holm(main_rows, cfg)
    versus = []
    zs_main_path = zsroot / "tables" / "zs_main.csv"
    if zs_main_path.exists():
        zs = pd.read_csv(zs_main_path)
        for row in comparisons.values():
            if row["label_view"] == "V7-VSMEC":
                continue  # Step2 has no zero-shot seven-class score to compare.
            matched = zs.loc[zs.backbone.eq(row["backbone"]) & zs.direction.eq("A" if row["direction"] == "SA" else "B") & zs.label_view.eq(row["label_view"])]
            if len(matched) == 1:
                versus.append(dict(backbone=row["backbone"], direction=row["direction"], label_view=row["label_view"], f1_base=row["base_f1"], f1_zs=matched.iloc[0].zs_f1,
                                   f1_seq=row["seq_f1"], gain=row["gain"], gap_step2=matched.iloc[0].gap))
    else:
        print("WARN: zs_main.csv missing; no cross-step comparison available")
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for filename, rows, columns in (("seq_main.csv", main_rows, MAIN_COLUMNS), ("seq_per_class.csv", classes, CLASS_COLUMNS), ("seq_c0.csv", controls, C0_COLUMNS), ("seq_vs_zs.csv", versus, VERSUS_COLUMNS)):
        table = pd.DataFrame(rows, columns=columns)
        table.to_csv(output / filename, index=False, lineterminator="\n")
        tables[filename] = table
    lr_table = root / "seq_lr_selection.csv"
    if lr_table.exists():
        (output / lr_table.name).write_bytes(lr_table.read_bytes())
    else:
        pd.DataFrame(columns=["backbone", "direction", "lr", "dev_macro_f1", "selected", "edge"]).to_csv(output / "seq_lr_selection.csv", index=False)
    for alias, direction, kind, counts, names in matrices:
        columns = names if kind != "7x3" else ["neg", "neu", "pos"]
        for suffix, values in (("counts", counts), ("rownorm", row_normalize(counts))):
            pd.DataFrame(values, index=names, columns=columns).to_csv(output / f"seq_confusion_{alias}_{direction}_{kind}_{suffix}.csv", lineterminator="\n")
    frozen_path = Path("docs/seq_primary_comparisons.md")
    try:
        frozen_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(frozen_path)], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        frozen_commit = ""
    frozen = re.search(r"```json\s*(.*?)\s*```", frozen_path.read_text(encoding="utf-8"), re.S)
    if not frozen or json.loads(frozen.group(1)) != cfg["primary"]:
        raise ValueError("Config primary differs from frozen document")
    if not frozen_commit:
        print("WARN: primary comparisons have not been committed; commit before real experiments")
    meta = dict(boot=boot, boot_seed=seed, ci=cfg["ci"], alpha=cfg["alpha"], code_version=_git_commit(),
                data_hash=sorted(metadata.data_hash.unique()), primary=cfg["primary"], primary_frozen_commit=frozen_commit,
                lr_selected=selection, missing=missing, baseline_provenance=cfg["baseline_provenance"])
    atomic_json(output / "seq_meta.json", meta)
    print("=== [SEQ ANALYZE DONE] ===")
    return {"tables": tables, "meta": meta, "out_dir": output}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("results-dir", "zs-results-dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--data-dir", default="data/clean")
    p.add_argument("--out-dir")
    p.add_argument("--config", dest="config_path", default="configs/seq.yaml")
    p.add_argument("--boot", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--allow-missing", action="store_true")
    analyze_results(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
