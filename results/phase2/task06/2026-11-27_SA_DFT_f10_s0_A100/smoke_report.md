# Task06 — GPU smoke test

- Commit: af12db20672b198accf724cbd26cdbf59b32d162
- Run ID: ViSoBERT_SA_dft_f10_s0_lr3e-05_p2v1_h9c9d1961
- Configuration: ViSoBERT / DFT / SA / train 10% / seed0 / LR3e-5 / BF16 strict.
- Data: train 2305, dev 3305; test not scored.
- GPU: NVIDIA A100-SXM4-80GB
- Dev macro-F1 V3: 0.6637240821785192
- Best epoch / step: 2.0 / 146
- Actual update steps: 292
- Train seconds: 46.45270280099999
- GPU hours: 0.01290352855583333
- Peak VRAM (MiB): 2979.6591796875
- Total / trainable params: 97568259 / 97568259
- Truncation rate: 0.00737527114967462
- Artifact hashes, dev probabilities, finite loss, frozen files, SKIP: PASS.
- Resume fields/best/history: PASS; GPU interruption and continuation: PASS

Smoke has no main encoder/head export; resume.pt contains best weights.
Trainer cost: peak VRAM covers the final fit invocation, not all interrupted processes.
train_seconds excludes model downloads, setup, ZIP and unsaved work after the interrupted checkpoint.
DFT GPU hours do not precisely estimate Seq/C0/LoRA cost.
Actual environment is recorded in environment.json and pip-freeze.txt; historical baselines remain unverified.
Review logs and measurements before Task07; no LR selection from test.
