# Repository instructions

## Active protocol

- This repository uses the v3.0 research workflow with the `seq-v1` amendment.
  For sequential work read `docs/seq_notes.md`, `configs/seq.yaml`,
  `tasks/AGENT_step3_seq.md` and `docs/seq_primary_comparisons.md` first.
- `configs/seq.yaml` is the executable configuration contract. Do not infer
  new settings from older task files. Change the protocol version, documentation
  and validation tests together when intentionally changing the experiment.
- `tasks/AGENT_step3_seq (1).md` is a redirect. Other historical Step1/M* task
  files describe past implementation work; they do not override seq-v1.
  `Quy_uoc_bat_buoc_cho_moi_model.md` is archived v2 documentation.
- Do not copy broken train_neu3/stage1 code from another branch. Sequential
  training lives in src/seq and uses the shared classifier/label constants.
- Configs in Git are not evidence of the actual historical baseline runs.
  Keep provenance unverified until the real configs/logs have been checked.

## Scientific invariants

- Preserve frozen data/clean, data/splits, manifests and HARD_MERGE_RULE.
- Sweep uses target train/dev only; test scoring follows best-dev selection.
  Test scores must never select LR, epoch or early stopping.
- Preserve default zero-shot bootstrap RNG behavior. Sequential pairing needs
  explicit matching ordered seed IDs. Holm always has four primary comparisons;
  do not shrink the family when data is missing.
- Strict VSMEC scores labels0,2; neutral predictions remain errors. Export
  strict per-class results only for negative and positive.
- Run IDs, checkpoint/artifact hashes and provenance must prevent silently
  reusing stale runs. Resume must preserve best weights and data/RNG cursor.
- Never launch research training, access Drive or download models as part of
  offline implementation/testing unless the user explicitly requests it.

## Validation

Windows: `.venv/Scripts/python.exe -X utf8 -m pytest -q`.
Use small local models/fake trainers for new tests, including resume and
train/dev/test separation. Keep old tests green. Do not claim CUDA/Drive or
historical baseline verification from CPU tests. Do not commit/push unless asked.
