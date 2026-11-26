"""Run a Phase 2 experiment (no training on import or --help)."""
import argparse
from pathlib import Path

import yaml

from src.phase2.run import run_phase2_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--backbone", choices=["ViSoBERT", "XLMR"])
    parser.add_argument("--direction", choices=["SA", "SB"])
    parser.add_argument("--method")
    parser.add_argument("--fraction", type=int, choices=[5, 10, 25, 50, 100])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--run-mode", choices=["sweep", "smoke", "main"])
    for name in ("clean-dir", "subsets-dir", "results-dir", "ckpt-out-dir", "pretrained-model-path",
                 "source-ckpt-root", "c0-stage1-run-dir", "selection-path"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--no-save-ckpt", action="store_true", default=None)
    parser.add_argument("--allow-non-strict-precision", action="store_true")
    parser.add_argument("--epochs-override", type=int)
    args = vars(parser.parse_args())
    nonstrict = args.pop("allow_non_strict_precision")
    config_path = args.pop("config")
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # Smoke examples reference the locked protocol instead of duplicating it.
    if "protocol_config" in document:
        config_path = str((Path(config_path).resolve().parent / document["protocol_config"]).resolve())
        defaults = document["run"]
    else:
        defaults = {}
    options = {**defaults, **{k: v for k, v in args.items() if v is not None}}
    if nonstrict:
        options["strict_precision_override"] = False
    result = run_phase2_experiment(config_path, **options)
    print(f"{result['run_id']}: {'SKIP' if result['skipped'] else 'complete'}")


if __name__ == "__main__":
    main()
