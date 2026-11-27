# Task07 — Báo cáo dò Learning Rate (12 lượt dev-only, `phase2-v1`)

- **Commit:** `edf4067751d72f84ad5e70485614f2d1e71b6d9c`
- **Profile môi trường:** `colab-task06-v1` (session `20261127T113132Z`)
- **GPU / Precision:** `NVIDIA L4` / `bf16` strict
- **Python / Torch / CUDA:** `3.13.18` / `2.14.1+cu130` / `13.0`
- **Thời gian setup & preflight (trước sweep):** `57.077s`
- **Tổng `train_seconds` (12 lượt):** `1045.90s` (`0.2905` GPU-giờ)
- **Peak VRAM cực đại:** SA = `3334.7 MiB`, SB = `2051.4 MiB`
- **Checkpoint nguồn Stage 1 (`vsmec`, seed 0):** `ViSoBERT_Stage1-base-emokeep-bf16_s0_lr5e-5` (`encoder_sha256=3cf5ebc839dca08f73e654a0f5090309c9aa1f515e6e20a9b3e29009a5300805`)
- **Checkpoint nguồn Stage 1 (`neu3`, seed 0):** `ViSoBERT_NEU3-base-emokeep-bf16_s0_lr3e-5` (`encoder_sha256=da379459f87d4404b09ee206c5a426f814d47f7f696f00b6a7a92e41affb3bb4`)
- **Provenance lịch sử của checkpoint nguồn Stage 1:** `unverified` (theo đúng quy ước `AGENTS.md`).

## 1. Learning rate được khóa (`selection_p1_lr.json`)

| Khóa | Chiều | Phương pháp | LR được chọn | Điểm `dev` chọn | Cờ `grid_edge` | `run_id` |
|---|---|---|---:|---:|---|---|
| `ViSoBERT/SA/dft` | `SA` | `dft` | `3e-05` | `0.716630` | `False` | `ViSoBERT_SA_dft_f10_s0_lr3e-05_p2v1_h09960405` |
| `ViSoBERT/SA/seq` | `SA` | `seq` | `2e-05` | `0.659187` | `False` | `ViSoBERT_SA_seq_f10_s0_lr2e-05_p2v1_had5c9c35` |
| `ViSoBERT/SB/dft` | `SB` | `dft` | `2e-05` | `0.595990` | `True` | `ViSoBERT_SB_dft_f10_s0_lr2e-05_p2v1_had5c9c35` |
| `ViSoBERT/SB/seq` | `SB` | `seq` | `3e-05` | `0.596595` | `True` | `ViSoBERT_SB_seq_f10_s0_lr3e-05_p2v1_h09960405` |

## 2. Kết quả chi tiết cả 12 lượt sweep (`fraction=10`, `seed=0`, `dev`-only)

| # | Chiều | Phương pháp | LR | View chọn | Điểm chọn `dev` | `V7-VSMEC` `dev` | Best ep / step | Tổng step | Early stop | `train_s` | Peak VRAM (MiB) |
|---:|---|---|---:|---|---:|---:|---|---:|---|---:|---:|
| 1 | `SA` | `dft` | `2e-05` | `V3-NEU` | `0.703655` | `—` | `4.0 / 292` | `438` | `True` | `86.65` | `2959.2` |
| 2 | `SA` | `dft` | `3e-05` | `V3-NEU` | `0.716630` | `—` | `8.0 / 584` | `730` | `True` | `144.32` | `2959.2` |
| 3 | `SA` | `dft` | `5e-05` | `V3-NEU` | `0.709521` | `—` | `8.0 / 584` | `730` | `True` | `145.48` | `2959.2` |
| 4 | `SA` | `seq` | `1e-05` | `V3-NEU` | `0.644007` | `—` | `7.0 / 511` | `657` | `True` | `141.99` | `3334.7` |
| 5 | `SA` | `seq` | `2e-05` | `V3-NEU` | `0.659187` | `—` | `8.0 / 584` | `730` | `True` | `157.74` | `3334.7` |
| 6 | `SA` | `seq` | `3e-05` | `V3-NEU` | `0.629395` | `—` | `4.0 / 292` | `438` | `True` | `94.52` | `3334.7` |
| 7 | `SB` | `dft` | `2e-05` | `V5-VSMEC` | `0.595990` | `0.493676` | `5.0 / 90` | `126` | `True` | `41.51` | `2051.4` |
| 8 | `SB` | `dft` | `3e-05` | `V5-VSMEC` | `0.571428` | `0.476323` | `4.0 / 72` | `108` | `True` | `41.42` | `2051.4` |
| 9 | `SB` | `dft` | `5e-05` | `V5-VSMEC` | `0.531535` | `0.445296` | `2.0 / 36` | `72` | `True` | `26.89` | `2051.4` |
| 10 | `SB` | `seq` | `1e-05` | `V5-VSMEC` | `0.572217` | `0.482860` | `10.0 / 180` | `180` | `False` | `65.37` | `2051.4` |
| 11 | `SB` | `seq` | `2e-05` | `V5-VSMEC` | `0.586957` | `0.528482` | `7.0 / 126` | `162` | `True` | `56.69` | `2051.4` |
| 12 | `SB` | `seq` | `3e-05` | `V5-VSMEC` | `0.596595` | `0.532045` | `4.0 / 72` | `108` | `True` | `43.32` | `2051.4` |

## 3. Kiểm định & Ghi chú bàn giao

- **Kiểm định độc lập:** 12 `completion.json`, 12 `provenance.json`, 18 dòng `dev` trong `phase2_runs.csv`, không có `predictions_test.csv`, xác suất hợp lệ, `SKIP`, tài nguyên đóng băng: **PASS**.
- **Ước lượng sơ bộ thời gian trainer cho Task 08 (26 lượt ở 10% trên cùng loại GPU `NVIDIA L4`):** ~`0.640` GPU-giờ (chưa tính thời gian chấm `test` và lưu checkpoint `main`).
- `train_seconds` không tính thời gian cài đặt môi trường hoặc tải mô hình; nếu có lượt resume thực tế thì `peak_vram_mb` phản ánh lần gọi `fit` cuối.
