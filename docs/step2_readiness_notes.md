# Ghi chú kiểm chứng & hướng dẫn hoàn tất Step 2 Data Readiness (`p1`)

Tài liệu này tổng hợp các quyết định đã chốt, kết quả tự kiểm chứng mới nhất theo `configs/step2_data.yaml` và `scripts/check_step2_data.py`, đề xuất xử lý `VSMEC-strict`, và danh sách chính xác các đầu vào người phụ trách cần cung cấp trước khi chạy **Bước 2** (Zero-shot `VSMEC ↔ NEU-ESC`).

---

## 1. Trạng thái tổng thể & các quyết định đã chốt

**Kết luận cổng hiện tại (`--with-tokenizer`, chưa truyền CSV Bước 1 và `--check-ckpt`):**

- `CHƯA SẴN SÀNG: E1, E2, E3, F1, F3, G1, G2`
- **Đã `PASS`:** `A1`–`A4`, `B1`–`B3`, `C1`–`C3`, `D1`–`D3`, `F2`, `H1`–`H4`, `I1`–`I4`.
- **`WARN` (`FAIL` đã biết / kiểm soát được):** `D4` (2 cặp near-duplicate), `F2b` (6 artifact cũ/ngoài phạm vi Bước 2 trong `data/clean/`).
- **`SKIP`:** `J1`–`J3` (chỉ chạy khi truyền `--check-ckpt` và đường dẫn checkpoint trên Drive/Colab).
- **`MISSING`:** `E1`, `E2`, `E3`, `F1`, `F3`, `G1`, `G2`, `G3` (`WARN`).

### Quyết định 1: Giữ nguyên `data/clean/MANIFEST.json`, tách `F2` (`BLOCKER`) và `F2b` (`WARN`)
- **`MANIFEST.json` KHÔNG cập nhật.** Toàn bộ `data/clean/`, `data/splits/` giữ nguyên trạng thái đóng băng.
- **Thu hẹp phạm vi `F2` (`BLOCKER`):** `F2` chỉ kiểm tra SHA-256 của 8 file Bước 2 thực sự sử dụng được khai báo trong `configs/step2_data.yaml` (`step2_files`):
  - `vsmec_train.csv`, `vsmec_dev.csv`, `vsmec_test.csv`, `neuesc_100.csv`, `vsmec_removed.csv`, `overlap_matrix.csv`, `internal_stats.csv`, `excluded_records.csv`.
  - Kết quả hiện tại: **8/8 file khớp SHA-256 trong `MANIFEST.json` (`F2 = PASS`)**.
- **Mục mới `F2b` (`WARN`):** Các file còn lại trong `data/clean/` ngoài phạm vi Bước 2 (bao gồm các artifact cũ của CV-P3 lấy mẫu từ `vsmec_train`) được ghi nhận ở `F2b`:

| File | SHA-256 thực tế | SHA-256 trong `MANIFEST.json` | Ghi chú |
| --- | --- | --- | --- |
| `manifest_run1.json` | `80a19181ad2c88373af0b93289c978328bed4569963acbb4a5a5a8d3d2fdde9c` | `không có` | Artifact ngoài phạm vi Bước 2 |
| `polarity_key.csv` | `4c5587afe006a905c71af1441b1125a0bcda1035786e66ec828a1a692201b15b` | `không có` | Artifact cũ CV-P3 (`vsmec_train`) |
| `polarity_long.csv` | `054ffc6808815a79ae9c3601d5e12977fbbadf6ba971d2e1e8ed17da2a05b0e6` | `bba9a68c2af398bdca44eccf6fea1194802f26b641fa1455c8ddbe7eddfc6834` | Lệch MANIFEST đã biết (CV-P3) |
| `polarity_nhat.csv` | `a22f50bcd290569e65e943b2e2c1c575b176762fd1941560458841d237df98da` | `bba9a68c2af398bdca44eccf6fea1194802f26b641fa1455c8ddbe7eddfc6834` | Lệch MANIFEST đã biết (CV-P3) |
| `vsfc_test_clean.csv` | `e6873a782eb1368b85a46fcabee091aa081a174f01fdc759951558aba4ed6023` | `e6873a782eb1368b85a46fcabee091aa081a174f01fdc759951558aba4ed6023` | Khớp MANIFEST, ngoài phạm vi Bước 2 |
| `vsmec_clean_report.json` | `7071ea3ec51d3a8a72e597e39225757b139a33003758e6f52400b94dd4b59dd8` | `không có` | Artifact ngoài phạm vi Bước 2 |

> Lưu ý: `polarity_long.csv`, `polarity_nhat.csv`, `polarity_key.csv` là artifact cũ của CV-P3 (lấy mẫu từ `vsmec_train`), **không** được Bước 2 sử dụng cho `VSMEC-strict`. Sai lệch với `MANIFEST.json` là hạn chế đã biết.

### Quyết định 2: Giữ nguyên dữ liệu đóng băng ở `D4`, xuất file kiểm tra độ nhạy
- **Không loại bất kỳ câu nào** khỏi `data/clean/vsmec_test.csv` hay `data/clean/neuesc_100.csv`.
- **Trùng chính xác (`D1`, `D2`, `D3`):** Trên cả 6 cặp `(test_split, ref_split)` theo 2 chiều (`3×2×2`) sau `normalize_text` + `hash_text`, số câu trùng = **`0` (`PASS`)**.
- **Trùng gần (`D4`, `WARN`)** sau chuẩn hóa mạnh (bỏ dấu câu & emoji, giữ chữ cái tiếng Việt + chữ số) phát hiện **2 cặp**:
  1. `vsmec_test` (`id=6311`, `Sadness` [1] → `negative` [0]) ↔ `neuesc_dev` (`id=[25960]`, `negative` [0] → `negative` [0]), **khớp nhãn 3c = `True`**, chuỗi: `"chạm đáy nỗi đau"`.
  2. `neuesc_test` (`id=28482`, `neutral` [1] → `neutral` [1]) ↔ `vsmec_train` (`id=[3607]`, `Other` [6] → `neutral` [1]), **khớp nhãn 3c = `True`**, chuỗi: `"vật chất quyết định ý thức"`.
- **Tỉ lệ ảnh hưởng độ nhạy:** `vsmec_test` = **1/692 (0.1445%)**; `neuesc_test` = **1/6613 (0.0151%)**.
- Đã xuất sẵn **`reports/step2/sensitivity_exclude_ids.csv`** chứa `vsmec_test id=6311` và `neuesc_test id=28482` để Bước 2 chạy thêm một lượt độ nhạy song song với lượt đầy đủ.

---

## 2. Kết quả kiểm tra chi tiết mới nhất (`H1`–`H4`, `I1`–`I4`, hàm băm Bước 1)

### 2.1. Phân bố nhãn & JS Divergence (`H1`–`H3`)
- **VSMEC 7 lớp (`train`/`dev`/`test` = 5,542 / 686 / 692):**
  - `test` (`n=692`): `Enjoyment`=193, `Sadness`=116, `Fear`=46, `Anger`=40, `Disgust`=131, `Surprise`=37, `Other`=129. Lớp nhỏ nhất trên `vsmec_test` là `Surprise` = 37 (≥ 30).
- **VSMEC 3 lớp (gộp theo `HARD_MERGE_RULE` B1) vs NEU-ESC 3 lớp (`train`/`dev`/`test` = 23,048 / 3,305 / 6,613):**
  - `train`: VSMEC (`neg=2723, neu=1261, pos=1558`, `n=5542`) vs NEU-ESC (`neg=4215, neu=15936, pos=2897`, `n=23048`) → `JS = 0.163287 bits` (`0.113182 nats`).
  - `dev`: VSMEC (`neg=301, neu=171, pos=214`, `n=686`) vs NEU-ESC (`neg=609, neu=2279, pos=417`, `n=3305`) → `JS = 0.145426 bits` (`0.100802 nats`).
  - `test`: VSMEC (`neg=333, neu=166, pos=193`, `n=692`) vs NEU-ESC (`neg=1221, neu=4558, pos=834`, `n=6613`) → `JS = 0.152440 bits` (`0.105664 nats`).
- **Hệ quả B1 trên `vsmec_test` (`H3`):** `Surprise` = 37 (5.35%), `Other` = 129 (18.64%), tổng gộp vào `neutral` = **166 / 692 (23.99%)**.

### 2.2. Lệch bề mặt giữa hai miền trên cả 6 split (`H4`)

| Bộ dữ liệu | Split | Số câu | Kết thúc `" ."` | Kết thúc `"."` | Có emoji Unicode | Số từ TB | Số từ p95 | Toàn chữ thường |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VSMEC | `train` | 5542 | 6.42% | 8.88% | 17.27% | 14.01 | 33.0 | 100.00% |
| VSMEC | `dev` | 686 | 5.39% | 7.87% | 16.62% | 14.18 | 31.0 | 100.00% |
| VSMEC | `test` | 692 | 6.65% | 9.10% | 19.08% | 13.80 | 31.0 | 100.00% |
| NEU-ESC | `train` | 23048 | 98.66% | 98.66% | 0.00% | 24.81 | 81.0 | 99.99% |
| NEU-ESC | `dev` | 3305 | 98.70% | 98.70% | 0.00% | 25.12 | 82.0 | 99.97% |
| NEU-ESC | `test` | 6613 | 98.53% | 98.53% | 0.00% | 26.02 | 89.0 | 100.00% |

Hiệu ứng này cần được xét ở phần "độ lệch nội tại" của Bước 2.

### 2.3. Độ dài token & tiền xử lý (`I1`–`I4`)
- **`I1` (`PASS`):** Cột `text_seg` đầy đủ, không rỗng/NaN trên cả `vsmec_test` (692) và `neuesc_test` (6,613).
- **`I2` (`PASS`, `--with-tokenizer`, ngưỡng `max_length = 256`):**
  - `vsmec_test`: PhoBERT = **0/692 (0.00%)**, ViSoBERT = **0/692 (0.00%)**.
  - `neuesc_test`: PhoBERT = **45/6613 (0.68%)**, ViSoBERT = **70/6613 (1.06%)**.
- **`I3` (`PASS`):** `normalize_text` bất biến (`idempotent`) trên toàn bộ 7,305 câu test (`692 + 6613`).
- **`I4` (`PASS`):** Đường dẫn ViSoBERT `emoji_mode='keep'` (`preprocess_text(..., emoji_mode='keep')` trên `text`) đồng nhất, bất biến và không sinh chuỗi rỗng.

### 2.4. Hàm băm dữ liệu & hỗ trợ nhiều schema kết quả Bước 1 (`F1`, `F3`, `G1`–`G3`)
- `vsmec_data_hash` (`compute_vsmec_data_hash(data/clean)`): `680bfc0f241ec6d4ef87e3e37c6d7552591779f40182b90c6654173c02960f95`.
- `neuesc_data_hash` (`sha256_file(data/clean/neuesc_100.csv)`): `9b0e3f42c2b7fa279e6887d15b522c566dd30d1c28430741972ed6d86ab03ac1`.
- `scripts/check_step2_data.py` hỗ trợ đọc gộp nhiều file qua `--results-file` (lặp được nhiều lần), `--results-glob`, hoặc mặc định quét `reports/step2/inputs/` (đã nằm trong `.gitignore`), loại dòng `sweep`, loại trùng theo `(run_id, split)`, và chịu được cả 3 schema: `STAGE1_COLUMNS`, `RESULT_COLUMNS`, `NEU3_COLUMNS`.

---

## 3. Đề xuất `scripts/make_strict_template.py` cho `VSMEC-strict` (`E1`–`E3`)

Hiện tại `E1`–`E3` đang ở trạng thái **`MISSING`** vì chưa chốt quy tắc `VSMEC-strict` và `vsmec_strict_file: null` trong `configs/step2_data.yaml`.
Script **`scripts/make_strict_template.py`** được tạo sẵn **chỉ như một đề xuất** (mặc định chạy dry-run, **không** ghi file ra đĩa trừ khi truyền `--generate`):
- Nếu chạy với `--generate`, script lấy mẫu ngẫu nhiên **100 câu** (với `seed=42`) trong tổng số **166 câu** thuộc lớp `Surprise` (5) và `Other` (6) của `data/clean/vsmec_test.csv`, xuất 4 cột `id, text, annotator_1, annotator_2` để gán nhãn cực tính độc lập (`0=neg, 1=neu, 2=pos`).

Người phụ trách cần chốt 1 trong 2 phương án sau và ghi vào `docs/vsmec_strict_rule.md`:
1. **Phương án 1 (rule-based strict):** Loại bỏ toàn bộ 166 câu `Surprise`/`Other` khỏi tập đánh giá `VSMEC-strict` (giữ lại **526 câu** thuộc 5 cảm xúc có cực tính rõ ràng trên `vsmec_test`). Không cần sinh phiếu gán nhãn tay.
2. **Phương án 2 (manual polarity annotation):** Chạy `.venv\Scripts\python.exe scripts/make_strict_template.py --generate` (hoặc gán toàn bộ 166 câu `Surprise`/`Other`), 2 người gán nhãn độc lập đạt Cohen's $\kappa \ge 0.60$, hòa giải tạo file CSV đích, và khai báo đường dẫn vào `vsmec_strict_file` trong `configs/step2_data.yaml`.

---

## 4. Danh sách chính xác những gì người phụ trách cần cung cấp để mở cổng Bước 2

Để chạy lại `scripts/check_step2_data.py` và đạt trạng thái `SẴN SÀNG`:

1. **File CSV kết quả Bước 1 (`F1`, `F3`, `G1`, `G3`):**
   - Tải `stage1.csv` (VSMEC) và `neu3.csv` (NEU-3) của **cả hai backbone** (`PhoBERT`, `ViSoBERT`) đặt vào thư mục `reports/step2/inputs/` (hoặc truyền `--results-file` / `--results-glob`).
   - Yêu cầu dữ liệu: mỗi nhóm `(backbone, miền)` trong 4 nhóm (`VSMEC PhoBERT`, `VSMEC ViSoBERT`, `NEU-3 PhoBERT`, `NEU-3 ViSoBERT`) có đủ **5 seed `0..4`** cho lượt `baseline` (không tính dòng `sweep`), và mỗi lượt có đủ cả `dev` lẫn `test`, thống nhất `data_hash` và `code_version`.
2. **Điểm chuẩn công bố trên VSMEC 7 lớp (`G2`):**
   - Điền 2 con số thực `macro-F1` vào `configs/step2_data.yaml`:
     - `vsmec_published_macro_f1.phobert`
     - `vsmec_published_macro_f1.visobert`
3. **Quyết định & dữ liệu `VSMEC-strict` (`E1`–`E3`):**
   - Chốt quy tắc dựng `VSMEC-strict` vào `docs/vsmec_strict_rule.md` (Phương án 1 hoặc Phương án 2).
   - Điền đường dẫn `vsmec_strict_file` trong `configs/step2_data.yaml` (cùng cột/file gán nhãn 2 người nếu áp dụng Phương án 2 để kiểm tra $\kappa \ge 0.60$).
4. **Đường dẫn 4 nhóm checkpoint Bước 1 trên Google Drive / local (`J1`–`J3`):**
   - Cung cấp 4 đường dẫn tương ứng với `--ckpt-root-vsmec-phobert`, `--ckpt-root-vsmec-visobert`, `--ckpt-root-neu3-phobert`, `--ckpt-root-neu3-visobert` (hoặc điền vào mục `ckpt_roots` của `configs/step2_data.yaml`), rồi chạy đoạn mã Colab ở Mục 5 của `reports/step2/readiness_report.md` với `--check-ckpt --reload-check` để kiểm tra tải lại trọng số và độ lệch logits FP32 ≤ `2e-3`.
