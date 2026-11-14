"""Sequential transfer CLI; no real runs are launched on import."""
import argparse
from src.seq.stage2 import run_stage2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", dest="config_path", default="configs/seq.yaml")
    p.add_argument("--backbone", choices=["PhoBERT", "ViSoBERT"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--direction", choices=["SA", "SB", "C0N", "C0V"], required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--variant", default="seq-bf16")
    for flag in ("vsmec-ckpt-root", "neu-ckpt-root", "data-dir", "results-dir", "ckpt-out-dir"):
        p.add_argument("--" + flag, **({"default": {"data-dir": "data/clean", "results-dir": "results", "ckpt-out-dir": "checkpoints/seq"}[flag]} if flag in ("data-dir", "results-dir", "ckpt-out-dir") else {}))
    p.add_argument("--max-examples", type=int)
    p.add_argument("--no-save-ckpt", action="store_true")
    run_stage2(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
