"""Paired seed/test contrasts and fixed-family multiple-comparison reporting."""
import numpy as np
from src.stats.bootstrap import bootstrap_contrast, weighted_statistics
from src.stats.holm import bootstrap_pvalue, holm_adjust


def paired_gain(base, seq, labels, seeds, *, boot=10000, seed=0, ci=0.95, stat="macro_f1"):
    if base["ids"].tolist() != seq["ids"].tolist() or not np.array_equal(base["y_true"], seq["y_true"]):
        raise ValueError("Paired arms must use identical ordered test IDs and labels")
    if base["seeds"] != seq["seeds"] or base["seeds"] != list(seeds):
        raise ValueError("Paired arms must use identical ordered seed IDs")
    arms = {name: {"y_true": arm["y_true"], "preds_by_seed": arm["preds"], "labels": labels, "seeds": arm["seeds"], "stat": stat}
            for name, arm in (("base", base), ("seq", seq))}
    estimate = bootstrap_contrast(arms, lambda x: x["seq"] - x["base"], True, share_seeds=True,
                                  boot=boot, boot_seed=seed, ci=ci)
    values = {name: weighted_statistics(arm["y_true"], arm["preds_by_seed"], labels, np.ones(len(arm["y_true"])), stat)[0] for name, arm in arms.items()}
    estimate.update(base_f1=float(values["base"].mean()), seq_f1=float(values["seq"].mean()),
                    base_std=float(values["base"].std(ddof=1)) if len(seeds) > 1 else 0.0,
                    seq_std=float(values["seq"].std(ddof=1)) if len(seeds) > 1 else 0.0,
                    n_seed_pos=int(np.count_nonzero(values["seq"] > values["base"])),
                    p_boot=bootstrap_pvalue(estimate["draws"]))
    return estimate


def apply_primary_holm(rows, config):
    keys = {(p["backbone"], p["direction"], p["label_view"], p["stat"]) for p in config["primary"]}
    selected = []
    for row in rows:
        row["primary"] = (row["backbone"], row["direction"], row["label_view"], row["stat"]) in keys
        row["exploratory"] = not row["primary"]
        row["p_holm"], row["significant"] = np.nan, None
        if row["primary"]:
            selected.append(row)
    actual_keys = {(r["backbone"], r["direction"], r["label_view"], r["stat"]) for r in selected}
    if len(selected) == 4 and actual_keys == keys and all(r.get("n_seeds", 5) == 5 for r in selected):
        for row, value in zip(selected, holm_adjust([r["p_boot"] for r in selected])):
            row["p_holm"] = float(value)
            row["significant"] = bool(value < config["alpha"])
    else:
        print("WARN: Holm withheld: require all four primary comparisons with five paired seeds")
    return rows
