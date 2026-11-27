"""Lock P1 learning rates using the complete dev-only sweep grid."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

from src.phase2.protocol import canonical_digest, load_phase2_config
from src.common.hashing import sha256_file
from src.phase2.models import artifact_path
from src.phase2.run import atomic_json
from src.zs.infer import _atomic_csv


def select_phase2_p1_lr(results_dir, out=None, config_path="configs/phase2.yaml",
                        backbone="ViSoBERT"):
    """Lock P1 learning rates from a complete dev-only sweep grid.

    Parameters
    ----------
    backbone : str
        ``"ViSoBERT"`` (default, Task 07) or ``"XLMR"`` (Task 11).
    """
    cfg = load_phase2_config(config_path)
    if backbone not in ("ViSoBERT", "XLMR"):
        raise ValueError(f"Unsupported backbone: {backbone}")
    root = Path(results_dir)
    frame = pd.read_csv(root / "phase2_runs.csv", keep_default_na=False)
    sweep = frame.loc[frame.run_mode.eq("sweep")]
    if not sweep.split.eq("dev").all():
        raise ValueError("test rows detected in sweep")
    grid = sweep.loc[sweep.backbone.eq(backbone) & sweep.fraction.eq(10) & sweep.seed.eq(0)
                     & sweep.method.isin(["dft", "seq"])].copy()
    if set(grid.protocol_version) != {cfg["protocol_version"]}:
        raise ValueError("Mixed/missing protocol_version in sweep")
    if "max_examples" in grid and not grid.max_examples.eq("").all():
        raise ValueError("Truncated sweep cannot select research LR")
    if grid.git_commit.nunique() != 1:
        raise ValueError("Mixed git_commit in sweep")
    suffix = "xlmr" if backbone == "XLMR" else "p1"
    selected, records, code_hashes = {}, [], set()
    for direction in ("SA", "SB"):
        domain = grid.loc[grid.direction.eq(direction)]
        if domain.data_hash.nunique() != 1 or domain.ids_sha256.nunique() != 1:
            raise ValueError("Mixed/missing data_hash or subset IDs in sweep")
        view = cfg["directions"][direction]["dev_selection_view"]
        for method in ("dft", "seq"):
            block = domain.loc[domain.method.eq(method) & domain.label_view.eq(view) & domain.map_method.eq("none")]
            expected = cfg["sweeps"][f"{method}_lr_grid"]
            if len(block) != len(expected) or sorted(block.lr.tolist()) != sorted(expected):
                raise ValueError(f"Incomplete/duplicate LR grid: {direction}/{method}")
            if not np.isfinite(block.dev_selection_score.to_numpy(float)).all():
                raise ValueError("Nonfinite dev_selection_score")
            for row in block.itertuples(index=False):
                run_dir = root / "runs" / row.run_id
                completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
                artifacts = completion.get("artifacts", {})
                required = {"config.yaml", "metadata.json", "metrics.json", "predictions_dev.csv", "cost.json"}
                if (completion.get("complete") is not True or not required.issubset(artifacts)
                    or (run_dir / "predictions_test.csv").exists()
                    or any(not artifact_path(run_dir, p).is_file() or sha256_file(artifact_path(run_dir, p)) != digest
                           for p, digest in artifacts.items())):
                    raise ValueError(f"Incomplete/tampered sweep artifacts: {row.run_id}")
                effective = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
                if effective["protocol_cfg"] != cfg or effective["settings"] != {**cfg["training"], "lr": row.lr}:
                    raise ValueError("Sweep does not use the locked training profile")
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                if metrics.get("test_metrics") is not None or not np.isclose(metrics["dev_selection_score"], row.dev_selection_score, rtol=0, atol=1e-12):
                    raise ValueError("Sweep score/test isolation mismatch")
                provenance = json.loads((root / "work" / row.run_id / "provenance.json").read_text(encoding="utf-8"))
                if provenance.get("fingerprint") != completion.get("fingerprint"):
                    raise ValueError("Sweep fingerprint mismatch")
                code_hashes.add(canonical_digest(provenance["source_code_sha256"]))
            winner = block.sort_values(["dev_selection_score", "lr"], ascending=[False, True]).iloc[0]
            edge = float(winner.lr) in (min(expected), max(expected))
            if edge:
                warnings.warn(f"{direction}/{method}: selected LR {winner.lr:g} is at grid edge", UserWarning)
            record = {"backbone": backbone, "direction": direction, "method": method,
                      "lr": float(winner.lr), "dev_selection_score": float(winner.dev_selection_score),
                      "data_hash": str(winner.data_hash), "ids_sha256": str(winner.ids_sha256),
                      "run_id": str(winner.run_id), "grid_edge": edge}
            selected[f"{backbone}/{direction}/{method}"] = record
            records.append(record)
    if len(code_hashes) != 1:
        raise ValueError("Mixed source code hashes in sweep")
    result = {"protocol_version": cfg["protocol_version"], "protocol_hash": canonical_digest(cfg),
              "tie_breaker": "smaller_lr", "selected": selected}
    out_name = f"selection_{suffix}_lr.json"
    atomic_json(Path(out) if out else root / out_name, result)
    _atomic_csv(pd.DataFrame(records), root / f"{suffix}_lr_selection.csv")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/phase2")
    parser.add_argument("--out")
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--backbone", choices=["ViSoBERT", "XLMR"], default="ViSoBERT")
    args = parser.parse_args()
    select_phase2_p1_lr(args.results_dir, args.out, args.config, backbone=args.backbone)


if __name__ == "__main__":
    main()
