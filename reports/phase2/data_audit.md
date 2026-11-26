# Báo cáo Task 02: Kiểm tra dữ liệu, tiền xử lý và giao thức cũ

**Kết luận cổng chuyển sang Task 03:** **SẴN SÀNG SANG TASK 03 (Tạo tập con dữ liệu)**

> Mọi kiểm tra chạy ở chế độ chỉ đọc (không chỉnh sửa bất kỳ file nào trong `data/` hay `configs/seq.yaml`). Chi tiết từng mã băm xem tại [manifest_audit.json](manifest_audit.json).

## 1. Kiểm toán `MANIFEST.json` và lỗi `polarity_long.csv` / `polarity_nhat.csv`

- `data/clean/MANIFEST.json`: trạng thái tổng **`WARN`** | 4 file lõi train/dev/test (`vsmec_{train,dev,test}.csv`, `neuesc_100.csv`): **khớp 100% (`core_files_ok=True`)**.
- `data/splits/MANIFEST.json`: trạng thái tổng **`PASS`** (`12` mục).
- File nằm trên đĩa `data/clean/` nhưng không có trong `MANIFEST.json`: `3` file (`manifest_run1.json`, `polarity_key.csv`, `vsmec_clean_report.json`).

### 1.1. Điều tra riêng `polarity_long.csv` và `polarity_nhat.csv`

| File | Kích thước (bytes) | Số dòng | Nhãn đã điền | SHA-256 trong `data/clean/MANIFEST.json` | SHA-256 thực tế trên đĩa | Khớp `results/polarity/polarity_files_hash.json`? |
|---|---:|---:|---:|---|---|---|
| `polarity_long.csv` | 6863 | 100 | 100 | `bba9a68c2af398bd...` | `054ffc6808815a79...` | **True** |
| `polarity_nhat.csv` | 6863 | 100 | 100 | `bba9a68c2af398bd...` | `a22f50bcd290569e...` | **True** |

- **Nguyên nhân:** Mã hash trùng trong data/clean/MANIFEST.json phản ánh ảnh chụp file mẫu trống trước khi hai người gán nhãn (Long, Nhật) điền cột polarity_label; sau khi gán nhãn, hash thật của từng file đã được lưu khớp tại results/polarity/polarity_files_hash.json nhưng data/clean/MANIFEST.json đóng băng không được ghi đè.
- **Ảnh hưởng đến Task 03:** KHÔNG ảnh hưởng đến việc tạo tập con train/dev/test ở Task 03 (hai file polarity chỉ phục vụ gán nhãn kiểm chứng M1b/M7, không tham gia nạp train/dev/test của UIT-VSMEC hay NEU-ESC).
- **Phương án đề xuất (chờ nhóm duyệt):** Giữ nguyên data/clean/MANIFEST.json đóng băng để không làm lệch provenance Giai đoạn 1; ghi nhận ngoại lệ này trong biên bản Task 02 và chỉ dùng results/polarity/polarity_files_hash.json khi đối chiếu artifact gán nhãn polarity.

## 2. Toàn vẹn dữ liệu Train / Dev / Test, Phân phối nhãn & Dự báo mức 5%

### 2.1. Kiểm tra cấu trúc, trùng lặp và rò rỉ

| Bộ dữ liệu | Schema & Nhãn | C1 (Trùng nội bộ) | C2 (Xung đột nhãn) | C3 (Rò rỉ giữa split) | D1/D2 (Trùng chéo miền trên test) |
|---|---|---:|---:|---:|---:|
| UIT-VSMEC (7 lớp) | `PASS` | 0 | 0 | 0 | 0 |
| NEU-ESC (3 lớp) | `PASS` | 0 | 0 | 0 | 0 |

- **Ghi chú D4 (trùng gần đã ghi nhận từ Bước 2):** Giữ dữ liệu đóng băng; 2 cặp trùng gần: vsmec_test id=6311 (Sadness→negative) ↔ neuesc_dev id=[25960] (negative→negative, khớp=True); neuesc_test id=28482 (neutral→neutral) ↔ vsmec_train id=[3607] (Other→neutral, khớp=True). Ảnh hưởng độ nhạy: vsmec_test 1/692 (0.1445%), neuesc_test 1/6613 (0.0151%) — đã xuất sensitivity_exclude_ids.csv
- **Quy mô `dev` UIT-VSMEC:** `686` câu (`>= 172`: **True** — nếu giữ nguyên full dev cố định ở Task 04 sẽ tự động thỏa ngưỡng tối thiểu `≥ 172` câu).

### 2.2. Phân bố nhãn UIT-VSMEC (7 lớp) & Dự báo phân tầng mức 5% (≈ 277 câu)

| Split | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 1558 | 947 | 316 | 391 | 1069 | 241 | 1020 | **5542** |
| `dev` | 214 | 86 | 31 | 49 | 135 | 30 | 141 | **686** |
| `test` | 193 | 116 | 46 | 40 | 131 | 37 | 129 | **692** |
| `train 5% (dự kiến)` | 78 | 47 | 16 | 20 | 53 | 12 | 51 | **277** |

- **Nhận xét sơ bộ mức 5% UIT-VSMEC:** Lớp nhỏ nhất dự kiến là `Surprise` với **`12` mẫu** (ngưỡng tối thiểu `< 5` mẫu để loại bỏ V7 → đạt ngưỡng kỹ thuật: **True**; bảng đếm chính thức trên 5 seed sẽ được xuất ở Task 03).

### 2.3. Phân bố nhãn NEU-ESC (3 lớp) & Khả năng truy vết nhãn `toxic` gốc (cho `V3-NEU-noToxic`)

| Split | negative (0) | neutral (1) | positive (2) | Tổng |
|---|---:|---:|---:|---:|
| `train` | 4215 | 15936 | 2897 | **23048** |
| `dev` | 609 | 2279 | 417 | **3305** |
| `test` | 1221 | 4558 | 834 | **6613** |

- **Truy vết nhãn gốc 4 lớp từ `data/raw/NEU_ESC_full.csv`:** **THÀNH CÔNG 100%** (khớp `id` = 32966/32966, khớp `normalize_text` = 100.00%, nhất quán ánh xạ `NEUESC_ORIG_TO_LABEL3` = `True`).

| Split (nhãn gốc 4 lớp) | neutral (0) | positive (1) | negative (2) | toxic (3) | Tổng |
|---|---:|---:|---:|---:|---:|
| `train` | 15936 | 2897 | 3630 | **585** | **23048** |
| `dev` | 2279 | 417 | 524 | **85** | **3305** |
| `test` | 4558 | 834 | 1046 | **175** | **6613** |
- **Kết luận `V3-NEU-noToxic`:** Tập `test` NEU-ESC có `175` câu gốc `toxic`; loại bỏ còn `6438` câu — **hoàn toàn khả thi tính lại ở Task 12** mà không cần sửa `neuesc_100.csv`.

## 3. Kiểm tra tiền xử lý ViSoBERT / XLM-R, tỷ lệ dấu `_` và tỷ lệ cắt cụt (`max_length=256`)

- **Trạng thái kiểm định tổng thể:** **`PASS`** (mọi split đều thỏa Unicode NFC, `normalize_text` idempotent, `preprocess_text(..., emoji_mode='keep')` idempotent và bảo toàn emoji).

| Tập dữ liệu | Số câu | Câu có emoji (gốc → keep) | Câu có `_` (gốc `text`) | Câu có `_` (đầu vào ViSoBERT/XLM-R) | Câu có `_` (`text_seg` PhoBERT) | Kiểm tra `_` | Hash đầu vào ViSoBERT/XLM-R (16 ký tự đầu) |
|---|---:|---|---:|---:|---:|---|---|
| `vsmec_train` | 5542 | 957 → 957 | 24 (0.43%) | 24 (0.43%) | 4406 (79.50%) | **PASS** | `e0d6c3dceaf7095d` |
| `vsmec_dev` | 686 | 114 → 114 | 1 (0.15%) | 1 (0.15%) | 535 (77.99%) | **PASS** | `dbc7d37ffc90ed7b` |
| `vsmec_test` | 692 | 132 → 132 | 3 (0.43%) | 3 (0.43%) | 555 (80.20%) | **PASS** | `d15cc0298e418561` |
| `neuesc_train` | 23048 | 0 → 0 | 0 (0.00%) | 0 (0.00%) | 17203 (74.64%) | **PASS** | `221929926b2d3330` |
| `neuesc_dev` | 3305 | 0 → 0 | 0 (0.00%) | 0 (0.00%) | 2484 (75.16%) | **PASS** | `5d122493e6fce342` |
| `neuesc_test` | 6613 | 0 → 0 | 0 (0.00%) | 0 (0.00%) | 4923 (74.44%) | **PASS** | `d3a0897f1d8783d4` |

### 3.1. Tỷ lệ cắt cụt ở `max_length = 256`

Số liệu đã ghi sẵn trong `results/token_stats.csv` (chỉ phủ **NEU-ESC**, chưa có bảng riêng cho **UIT-VSMEC**):

| Tokenizer | Split | Số câu | p95 token | Max token | Số câu cắt cụt (>256) | Tỷ lệ cắt cụt |
|---|---|---:|---:|---:|---:|---:|
| `PhoBERT (vinai/phobert-base-v2)` | `all` | 32966 | 76.0 | 1431 | 199 | 0.6037% |
| `PhoBERT (vinai/phobert-base-v2)` | `train` | 23048 | 74.65 | 1303 | 134 | 0.5814% |
| `PhoBERT (vinai/phobert-base-v2)` | `dev` | 3305 | 74.0 | 670 | 20 | 0.6051% |
| `PhoBERT (vinai/phobert-base-v2)` | `test` | 6613 | 81.4 | 1431 | 45 | 0.6805% |
| `ViSoBERT (uitnlp/visobert)` | `all` | 32966 | 89.0 | 1629 | 291 | 0.8827% |
| `ViSoBERT (uitnlp/visobert)` | `train` | 23048 | 87.0 | 1629 | 192 | 0.8330% |
| `ViSoBERT (uitnlp/visobert)` | `dev` | 3305 | 88.0 | 842 | 30 | 0.9077% |
| `ViSoBERT (uitnlp/visobert)` | `test` | 6613 | 96.0 | 1610 | 69 | 1.0434% |

- **Đo tokenizer trực tiếp (`--with-tokenizer`):** chế độ `SKIP_OFFLINE`.
  - Bỏ qua nạp tokenizer trực tiếp khi chạy mặc định offline (truyền --with-tokenizer để đo từ cache local); đã trích xuất số liệu NEU-ESC sẵn có từ results/token_stats.csv (chú ý results/token_stats.csv chưa có dòng riêng cho UIT-VSMEC).

## 4. Đối chiếu tính tương thích của Mốc 100% cũ với Giao thức Giai đoạn 2

- **Provenance trong `configs/seq.yaml`:** `status=repository_config_only`, `actual_run_verified=False`.

| Tiêu chí | Baseline trong Git (`base.yaml`) | Giao thức `seq-v1` (`configs/seq.yaml`) | Giao thức GĐ2 đề xuất | Đánh giá tương thích |
|---|---|---|---|---|
| Backbone phạm vi chính | vinai/phobert-base-v2 & uitnlp/visobert | PhoBERT, ViSoBERT | ViSoBERT (chính, 2 chiều); PhoBERT giữ số liệu 100% làm mốc; XLM-R & LLM ở mức 10% | COMPATIBLE (ViSoBERT giữ nguyên kiến trúc) |
| Tiền xử lý ViSoBERT & max_length | emoji_mode=tag (base.yaml cũ); max_length=256 | text_col=text, emoji_mode=keep, max_length=256 | text (văn bản thô, không '_'), emoji_mode=keep, max_length=256 | COMPATIBLE với seq-v1 & zs.yaml (cần xác minh log baseline Stage 1 thật có bật emokeep hay không) |
| Batch hiệu dụng / Epochs / Weight decay | 16×2=32, epochs=10 | 16×2=32, epochs=10, wd=0.01 | Batch hiệu dụng 32, tối đa 10 epoch (LoRA có thể nâng 15 nếu chưa hội tụ), weight_decay=0.01 | COMPATIBLE |
| Warmup ratio | Không khai báo tường minh trong base.yaml (nhánh Neu3 dùng 0.06) | 0.06 (6%) | 0.10 (10%) | DIFFERS (0.06 vs 0.10 — cần chốt ở Task 04) |
| Early stopping patience & tần suất eval | patience=5, metric=val_loss | patience=5, evals_per_epoch=2 | patience=2 (cần chốt evals_per_epoch ở Task 04) | DIFFERS (patience 5 vs 2 — cần chốt ở Task 04) |
| Tiêu chí chọn best checkpoint trên dev | val_loss (base.yaml) / macro_f1 V7 hoặc V3 trên nhánh train | Macro-F1 toàn bộ lớp view chính (V7-VSMEC cho SB; V3-NEU cho SA) | Đề xuất: Macro-F1 5 lớp (bỏ Surprise/Other) cho VSMEC, V3-NEU cho NEU-ESC; giữ dev ≥ 172 câu | DIFFERS trên chiều B (V7 vs Macro-F1 5 lớp — cần chốt ở Task 04) |
| Đối chứng C0 | Chưa có | c0_seeds=[0, 1, 2], c0_lr_mode=reuse; nạp checkpoint 100% cùng miền rồi train tiếp trên 100% | Đề xuất Task 04: C0 low-resource cả 2 pha chỉ được thấy tập con đích (10/25/50%) cùng seed; không nạp checkpoint 100% đích | DIFFERS ở mức <100% (bắt buộc sửa để không rò rỉ dữ liệu 100% đích) |
| Precision & Seed & Thống kê chính | precision=fp16 | precision=bf16 (strict=True), seeds=[0, 1, 2, 3, 4], Holm m=4 | precision=bf16, seeds=0..4 (C0: 0..2), Holm m=2 cho Seq−DFT ở 10% hai chiều (ViSoBERT) | COMPATIBLE (chú ý nâng cấp Holm sang m=2 cho GĐ2 ở file phân tích riêng) |

**Kết luận khả năng tái sử dụng mốc 100%:** CHƯA THỂ KẾT LUẬN TƯƠNG THÍCH HOÀN TOÀN TẠI TASK 02: (1) Trong workspace local chưa có file kết quả/log thật và dự đoán từng mẫu mức 100% của ViSoBERT/PhoBERT (baseline_provenance.actual_run_verified=false); (2) Có 3 điểm khác biệt giao thức giữa `seq-v1` (mốc 100%) và kế hoạch GĐ2 đề xuất: `warmup_ratio` (0.06 vs 0.10), `patience` (5 vs 2), và tiêu chí chọn best dev trên UIT-VSMEC (Macro-F1 V7 vs Macro-F1 5 lớp). Nhóm cần quyết định ở Task 04: giữ giao thức chọn dev/warmup/patience tương thích mốc 100%, hay dùng giao thức mới và trình bày tách biệt / chạy lại mốc 100% ViSoBERT.

## 5. Rà soát & Phương án sửa Notebook Probe (Other/Surprise)

- **Trạng thái file notebook Probe tại local:** **`MISSING`** — Chưa có notebook Probe (Other/Surprise) trong workspace local hoặc chưa cấu hình `phase1_external.probe_notebook_path`.
- **Checklist bản vá bắt buộc áp dụng ngay khi đưa notebook Probe từ Colab/Drive về (trước khi trích số vào báo cáo):**
  1. Đổi `PRIOR` từ phân phối nhãn của tập `test` sang phân phối nhãn của tập **`train`** (ví dụ tính từ `data/clean/vsmec_train.csv` gộp 3 nhãn hoặc `data/clean/neuesc_100.csv` lọc `subset == 'train'`).
  2. Loại bỏ nhánh `except` fallback ngầm sang `uitnlp/visobert`; bắt buộc dùng `AutoTokenizer.from_pretrained(Path(ckpt_dir) / 'encoder', local_files_only=True)` và raise lỗi ngay nếu thiếu file tokenizer.
  3. Khi đối chiếu hash file polarity, đọc `results/polarity/polarity_files_hash.json` thay vì mã hash mẫu trống cũ trong `data/clean/MANIFEST.json`.
