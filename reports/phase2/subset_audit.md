# Báo cáo Task 03: Kiểm định tập con phân tầng lồng nhau (Phase 2)

**Kết luận cổng Task 03:** **PASS (ĐẠT TOÀN BỘ 5 CỔNG S1–S5 — SẴN SÀNG SANG TASK 04 & TASK 05)**

> Sinh tự động từ `scripts/make_phase2_subsets.py` (`phase2-subsets-v1`, RNG `PCG64([20261126, dataset_code, seed, class_id])`). Dữ liệu đóng băng `data/clean/` và `data/splits/` giữ nguyên 100%; `dev` và `test` cố định không rút mẫu.

## 1. Tổng hợp 5 cổng kiểm định bắt buộc (`[S1]–[S5]`)

| Cổng | Nội dung kiểm định | Trạng thái | Số vi phạm |
|---|---|---|---:|
| `S1_nested_inclusion` | Tính lồng nhau nghiêm ngặt `5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%` (tổng ID & từng lớp, 5 seed) | **PASS** | 0 |
| `S2_zero_split_leakage` | 0 giao ID và `text_hash` với `dev`/`test`; giữ nguyên kích thước `dev`/`test` | **PASS** | 0 |
| `S3_stratification_drift_1pp` | Lệch phân phối từng lớp tối đa `≤ 1,0` điểm phần trăm ở `10%, 25%, 50%` | **PASS** | 0 |
| `S4_seed_diversity_and_identity` | 5 seed đôi một khác nhau; mọi phương pháp gọi loader trả về cùng tập ID | **PASS** | 0 |
| `S5_manifest_and_5pct_table` | Đủ 40 file chỉ mục khớp SHA-256 trong `MANIFEST.json` & bảng đếm 5% UIT-VSMEC | **PASS** | 0 |

## 2. Quy mô mẫu thực tế & Độ lệch phân phối lớp tối đa (so với `train` 100%)

| Bộ dữ liệu | Mức | Số câu `train` | `dev` cố định | `test` cố định | Lệch phân phối lớp lớn nhất (`|Δ%|`) | Đạt `≤ 1,0` điểm %? |
|---|---|---:|---:|---:|---:|---|
| UIT-VSMEC (7 lớp) | `5%` | **277** | 686 | 692 | 0.1650 pp | **PASS** |
| UIT-VSMEC (7 lớp) | `10%` | **554** | 686 | 692 | 0.1062 pp | **PASS** |
| UIT-VSMEC (7 lớp) | `25%` | **1386** | 686 | 692 | 0.0259 pp | **PASS** |
| UIT-VSMEC (7 lớp) | `50%` | **2771** | 686 | 692 | 0.0180 pp | **PASS** |
| UIT-VSMEC (7 lớp) | `100%` | **5542** | 686 | 692 | 0.0000 pp | **PASS** |
| NEU-ESC (3 lớp) | `5%` | **1152** | 3305 | 6613 | 0.0454 pp | **PASS** |
| NEU-ESC (3 lớp) | `10%` | **2305** | 3305 | 6613 | 0.0233 pp | **PASS** |
| NEU-ESC (3 lớp) | `25%` | **5762** | 3305 | 6613 | 0.0043 pp | **PASS** |
| NEU-ESC (3 lớp) | `50%` | **11524** | 3305 | 6613 | 0.0043 pp | **PASS** |
| NEU-ESC (3 lớp) | `100%` | **23048** | 3305 | 6613 | 0.0000 pp | **PASS** |

## 3. Phân bố từng lớp theo mức (tại `seed 0`, đồng nhất số lượng trên cả 5 seed)

### 3.1. UIT-VSMEC (`train`, 7 lớp cảm xúc)

| Mức | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `5%` | 78 | 47 | 16 | 20 | 53 | 12 | 51 | **277** |
| `10%` | 156 | 95 | 31 | 39 | 107 | 24 | 102 | **554** |
| `25%` | 390 | 237 | 79 | 98 | 267 | 60 | 255 | **1386** |
| `50%` | 779 | 474 | 158 | 195 | 535 | 120 | 510 | **2771** |
| `100%` | 1558 | 947 | 316 | 391 | 1069 | 241 | 1020 | **5542** |

### 3.2. NEU-ESC (`train`, 3 nhãn cực tính)

| Mức | negative (0) | neutral (1) | positive (2) | Tổng |
|---|---:|---:|---:|---:|
| `5%` | 211 | 796 | 145 | **1152** |
| `10%` | 421 | 1594 | 290 | **2305** |
| `25%` | 1054 | 3984 | 724 | **5762** |
| `50%` | 2108 | 7968 | 1448 | **11524** |
| `100%` | 4215 | 15936 | 2897 | **23048** |

## 4. Bảng đếm từng lớp mức 5% của UIT-VSMEC trên 5 seed (`vsmec_5pct_class_counts.csv`)

| Seed | Tổng | Enjoyment | Sadness | Fear | Anger | Disgust | Surprise | Other | Lớp nhỏ nhất | Số mẫu min | Đạt ngưỡng `≥ 5`? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `s0` | 277 | 78 | 47 | 16 | 20 | 53 | 12 | 51 | `Surprise` | **12** | **True** |
| `s1` | 277 | 78 | 47 | 16 | 20 | 53 | 12 | 51 | `Surprise` | **12** | **True** |
| `s2` | 277 | 78 | 47 | 16 | 20 | 53 | 12 | 51 | `Surprise` | **12** | **True** |
| `s3` | 277 | 78 | 47 | 16 | 20 | 53 | 12 | 51 | `Surprise` | **12** | **True** |
| `s4` | 277 | 78 | 47 | 16 | 20 | 53 | 12 | 51 | `Surprise` | **12** | **True** |

- **Đánh giá điều kiện mức 5% UIT-VSMEC (V7):** Lớp nhỏ nhất trên toàn bộ 5 seed là `Surprise` với **`12` mẫu** (`≥ 5` mẫu → **đủ điều kiện kỹ thuật về số mẫu tối thiểu: `True`**).
- **Chuyển tiếp sang Task 04:** Mức `5%` của cả UIT-VSMEC (`277` câu) và NEU-ESC (`1.152` câu) vẫn là hạng mục **tùy chọn theo ngân sách GPU** (+26 cấu hình lõi), quyết định bật/tắt chính thức tại **Task 04**.
- **Chuyển tiếp sang Task 05:** Thành viên 2 sử dụng trực tiếp hàm `src.phase2.subsets.load_phase2_target_splits(dataset, fraction, seed)` đọc từ `data/subsets_phase2/` + `MANIFEST.json`.
