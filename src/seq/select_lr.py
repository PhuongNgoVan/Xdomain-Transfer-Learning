"""Select only complete seed0 dev sweeps, never test scores."""
import argparse
from pathlib import Path

import pandas as pd

from src.seq.protocol import atomic_json, digest, load_config


def select_lr(results_dir, out=None, config_path="configs/seq.yaml"):
    cfg = load_config(config_path)
    frame = pd.read_csv(Path(results_dir) / "seq.csv", keep_default_na=False)
    dev = frame.loc[frame["split"].eq("dev") & frame["variant"].str.startswith("seqsweep-") & frame["seed"].eq(cfg["sweep_seed"])].copy()
    selected, table = {}, []
    for alias in cfg["backbones"]:
        for direction, view in (("SA", "V3-NEU"), ("SB", "V7-VSMEC")):
            rows = dev.loc[dev.backbone.eq(alias) & dev.direction.eq(direction) & dev.label_view.eq(view) & dev.map_method.eq("none")]
            if len(rows) != len(cfg["lr_grid"]) or set(rows.lr.astype(float)) != set(cfg["lr_grid"]):
                raise ValueError(f"Incomplete/duplicate LR grid for {alias}/{direction}")
            if rows.data_hash.nunique() != 1 or rows.code_version.nunique() != 1:
                raise ValueError("Cannot mix data/code versions within a sweep")
            scores = pd.to_numeric(rows.macro_f1, errors="raise")
            if not scores.between(0, 1).all():
                raise ValueError("Invalid dev macro-F1")
            rows = rows.assign(macro_f1=scores).sort_values(["macro_f1", "lr"], ascending=[False, True])
            lr = float(rows.iloc[0].lr)
            edge = lr in (min(cfg["lr_grid"]), max(cfg["lr_grid"]))
            if edge:
                print(f"WARN: selected LR on grid edge: {alias}/{direction}={lr}")
            selected[f"{alias}/{direction}"] = {"lr": lr, "dev_f1_by_lr": {str(float(r.lr)): float(r.macro_f1) for r in rows.itertuples()}, "rule": "max_dev_macro_f1_then_min_lr"}
            for r in rows.itertuples():
                table.append(dict(backbone=alias, direction=direction, lr=r.lr, dev_macro_f1=r.macro_f1, selected=float(r.lr) == lr, edge=edge and float(r.lr) == lr))
    payload = {"protocol_hash": digest(cfg["training"]), "selected": selected}
    atomic_json(out or Path(results_dir) / "selection_seq.json", payload)
    pd.DataFrame(table).to_csv(Path(results_dir) / "seq_lr_selection.csv", index=False, lineterminator="\n")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out")
    p.add_argument("--config", dest="config_path", default="configs/seq.yaml")
    select_lr(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
