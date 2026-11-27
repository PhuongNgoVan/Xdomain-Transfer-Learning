# Task08 — Báo cáo Thực nghiệm Chính mức 10% (`26` lượt `main`, `phase2-v1`)

- **Commit:** `edf4067751d72f84ad5e70485614f2d1e71b6d9c`
- **Profile môi trường:** `colab-task06-v1` (session `20261127T154606Z`)
- **GPU / Precision:** `NVIDIA A100-SXM4-80GB` / `bf16` strict
- **Python / Torch / CUDA:** `3.13.18` / `2.14.1+cu130` / `13.0`
- **Tổng `train_seconds` (26 lượt trainer):** `1583.88s` (`0.4400` GPU-giờ)
- **Peak VRAM cực đại:** SA = `3368.0 MiB`, SB = `2254.9 MiB`
- **Provenance lịch sử checkpoint nguồn Stage 1 (`vsmec` & `neu3`, `s0..4`):** `unverified` (theo `AGENTS.md`).
- **Ghi chú biên lưới (`grid_edge=True`) từ Task 07:** `SB/dft` (`2e-5`, cận dưới) và `SB/seq` (`3e-5`, cận trên) giữ nguyên theo `phase2-v1` và `selection_p1_lr.json`; không sửa LR sau khi chấm `test`.

## 1. Gia đình kiểm định xác nhận chính (`Holm m = 2`, `α = 0.05`) & Phân rã `C0`

| Comparison ID | Chiều | Đối chứng | View | Seeds | Ref F1 | New F1 | `Δ` (95% CI) | `p_boot` | `p_holm` | Kết luận (`m=2`) |
|---|---|---|---|---|---:|---:|---|---:|---:|---|
| `P2_SA_10` | `SA` | `seq_minus_dft` | `V3-NEU` | `0,1,2,3,4` | `0.7080` | `0.6338` | `-0.0743` (`[-0.0883, -0.0605]`) | `0.00020` | `0.00040` | `True` |
| `P2_SB_10` | `SB` | `seq_minus_dft` | `V7-VSMEC` | `0,1,2,3,4` | `0.5050` | `0.5137` | `+0.0087` (`[-0.0238, +0.0417]`) | `0.59114` | `0.59114` | `False` |
| `EXP_SA_10_SEQ_MINUS_C0` | `SA` | `seq_minus_c0` | `V3-NEU` | `0,1,2` | `0.7103` | `0.6431` | `-0.0672` (`[-0.0806, -0.0538]`) | `0.00020` | `— (exploratory)` | `exploratory` |
| `EXP_SA_10_C0_MINUS_DFT` | `SA` | `c0_minus_dft` | `V3-NEU` | `0,1,2` | `0.7103` | `0.7103` | `-0.0001` (`[-0.0079, +0.0081]`) | `0.97330` | `— (exploratory)` | `exploratory` |
| `EXP_SB_10_SEQ_MINUS_C0` | `SB` | `seq_minus_c0` | `V7-VSMEC` | `0,1,2` | `0.4981` | `0.5231` | `+0.0250` (`[-0.0127, +0.0630]`) | `0.17998` | `— (exploratory)` | `exploratory` |
| `EXP_SB_10_C0_MINUS_DFT` | `SB` | `c0_minus_dft` | `V7-VSMEC` | `0,1,2` | `0.5039` | `0.4981` | `-0.0058` (`[-0.0263, +0.0146]`) | `0.55594` | `— (exploratory)` | `exploratory` |

## 2. Tóm tắt kết quả `test` chính (`mean ± SD` trên 5/3 seeds)

| Chiều | Phương pháp | Seeds | View | `Macro-F1` | `Weighted-F1` | `Accuracy` |
|---|---|---:|---|---:|---:|---:|
| `SA` | `dft` | `5` | `V3-NEU/none` | `0.7080 ± 0.0046` | `0.7908 ± 0.0073` | `0.7924 ± 0.0111` |
| `SA` | `c0` | `3` | `V3-NEU/none` | `0.7103 ± 0.0022` | `0.7960 ± 0.0087` | `0.8002 ± 0.0146` |
| `SA` | `seq` | `5` | `V3-NEU/none` | `0.6338 ± 0.0140` | `0.7425 ± 0.0115` | `0.7467 ± 0.0129` |
| `SB` | `dft` | `5` | `V7-VSMEC/none` | `0.5050 ± 0.0216` | `0.5667 ± 0.0151` | `0.5798 ± 0.0106` |
| `SB` | `dft` | `5` | `V5-VSMEC/none` | `0.5512 ± 0.0127` | `0.6027 ± 0.0075` | `0.5798 ± 0.0106` |
| `SB` | `dft` | `5` | `V3-VSMEC/sum` | `0.6677 ± 0.0096` | `0.7007 ± 0.0059` | `0.7055 ± 0.0026` |
| `SB` | `dft` | `5` | `V3-VSMEC-strict/sum` | `0.8079 ± 0.0103` | `0.8197 ± 0.0093` | `0.7833 ± 0.0183` |
| `SB` | `c0` | `3` | `V7-VSMEC/none` | `0.4981 ± 0.0272` | `0.5605 ± 0.0210` | `0.5751 ± 0.0159` |
| `SB` | `c0` | `3` | `V5-VSMEC/none` | `0.5551 ± 0.0145` | `0.6059 ± 0.0139` | `0.5751 ± 0.0159` |
| `SB` | `c0` | `3` | `V3-VSMEC/sum` | `0.6580 ± 0.0189` | `0.6932 ± 0.0152` | `0.6970 ± 0.0159` |
| `SB` | `c0` | `3` | `V3-VSMEC-strict/sum` | `0.8113 ± 0.0122` | `0.8196 ± 0.0129` | `0.7814 ± 0.0198` |
| `SB` | `seq` | `5` | `V7-VSMEC/none` | `0.5137 ± 0.0179` | `0.5838 ± 0.0125` | `0.5948 ± 0.0127` |
| `SB` | `seq` | `5` | `V5-VSMEC/none` | `0.5793 ± 0.0134` | `0.6306 ± 0.0123` | `0.5948 ± 0.0127` |
| `SB` | `seq` | `5` | `V3-VSMEC/sum` | `0.6865 ± 0.0091` | `0.7185 ± 0.0054` | `0.7263 ± 0.0049` |
| `SB` | `seq` | `5` | `V3-VSMEC-strict/sum` | `0.8285 ± 0.0176` | `0.8418 ± 0.0160` | `0.8046 ± 0.0288` |

## 3. Kiểm định & Dự toán bước tiếp theo (Task 09)

- **Kiểm định độc lập (Bước 9):** 26 `completion.json`, 26 `provenance.json`, 130 dòng `phase2_runs.csv` (`39` `dev` + `91` `test`), khớp 100% thứ tự `test` ID, `C0 low-resource` sạch, `SKIP`, tài nguyên đóng băng: **PASS**.
- **Quy tắc `V3-VSMEC-strict` per-class (`AGENTS.md`):** `task08_per_class_f1.csv` chỉ xuất 2 lớp `negative` và `positive` (không xuất dòng `neutral`).
- **Dự toán thô thời gian trainer cho Task 09 (52 lượt ở 25% và 50% trên cùng GPU `NVIDIA A100-SXM4-80GB`):** ~`3.30` GPU-giờ (chỉ mang tính định hướng sơ bộ vì tỷ lệ early-stopping có thể thay đổi ở 25%–50%).
