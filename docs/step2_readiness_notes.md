# Ghi chú kiểm chứng & câu hỏi mở cho Step 2 Data Readiness (`p1`)

Tài liệu này tổng hợp kết quả tự kiểm chứng theo **Mục 5** của `tasks/AGENT_step2_data_readiness.md` và các câu hỏi cần xác nhận từ người hướng dẫn trước khi chạy **Bước 2** (Zero-shot `VSMEC ↔ NEU-ESC`).

---

## 1. Kiểm chứng cấu trúc thư mục `data/clean/` (Mục 5.1)

Toàn bộ các file trong `data/clean/` đã được kiểm tra thực tế (không chỉnh sửa bất kỳ file nào):

| Đường dẫn file | Số dòng | Danh sách cột chính xác | Ghi chú |
|---|---:|---|---|
| `data/clean/vsmec_train.csv` | 5,542 | `['id', 'text', 'label', 'subset', 'text_seg']` | Tập train VSMEC (`subset == "train"`), nhãn số `0..6`. |
| `data/clean/vsmec_dev.csv` | 686 | `['id', 'text', 'label', 'subset', 'text_seg']` | Tập dev VSMEC (`subset == "dev"`), nhãn số `0..6`. |
| `data/clean/vsmec_test.csv` | 692 | `['id', 'text', 'label', 'subset', 'text_seg']` | Tập test VSMEC (`subset == "test"`), nhãn số `0..6`. |
| `data/clean/neuesc_100.csv` | 32,966 | `['id', 'text', 'label', 'subset', 'text_seg']` | Toàn bộ NEU-ESC (`train`: 23,048; `dev`: 3,305; `test`: 6,613), nhãn số `0..2`. |
| `data/clean/neuesc_20.csv` | 6,609 | `['id', 'text', 'label', 'subset', 'text_seg']` | Bản con 20% NEU-ESC. |
| `data/clean/neuesc_5.csv` | 1,651 | `['id', 'text', 'label', 'subset', 'text_seg']` | Bản con 5% NEU-ESC. |
| `data/clean/vsfc_100.csv` | 16,175 | `['id', 'text', 'label', 'subset', 'text_seg']` | Không dùng trong v3.0 (theo quy ước phạm vi). |
| `data/clean/vsfc_20.csv` | 3,234 | `['id', 'text', 'label', 'subset', 'text_seg']` | Không dùng trong v3.0. |
| `data/clean/vsfc_5.csv` | 808 | `['id', 'text', 'label', 'subset', 'text_seg']` | Không dùng trong v3.0. |
| `data/clean/overlap_excluded.csv` | 18 | `['source', 'id', 'text', 'label', 'matched_corpus', 'matched_id']` | Danh sách câu bị loại khi chạy `src/data/check_overlap.py`. |
| `data/clean/polarity_nhat.csv` | 100 | `['id', 'text', 'vsmec_label', 'vsmec_label_name', 'polarity']` | 100 câu lấy mẫu từ `vsmec_train.csv` (50 `Surprise` + 50 `Other`) cho CV-P3 (`W_trans`). |
| `data/clean/polarity_long.csv` | 100 | `['id', 'text', 'vsmec_label', 'vsmec_label_name', 'polarity']` | File gán nhãn độc lập của annotator 2 (CV-P3). |
| `data/clean/polarity_key.csv` | 100 | `['id', 'text', 'vsmec_label', 'vsmec_label_name', 'polarity_nhat', 'polarity_long', 'polarity_final', 'adjudication_note']` | File chốt nhãn sau hòa giải (CV-P3) để ước lượng hàng `Surprise`/`Other` của `W_trans`. |
| `data/clean/MANIFEST.json` | — | JSON `{file: {sha256, bytes}}` | Ghi nhận SHA-256 của các file trong `data/clean/`. |

---

## 2. Hàm băm dữ liệu (`data_hash`) ở Bước 1.1 và Bước 1.2 (Mục 5.2)

Qua đối chiếu trực tiếp mã nguồn trên các nhánh `phobert-vsmec-7c`, `visobert-vsmec-7c`, `Neu3-phobert`, `Neu3-visobert` và `src/common/hashing.py`:

1. **Bước 1.1 (`src/plm/stage1.py` — VSMEC 7 lớp):**
   - Hàm sử dụng: `compute_vsmec_data_hash(data_dir)` trong `src/common/hashing.py` (hoặc truyền qua `--data-hash`).
   - Công thức: tính `sha256_file` cho từng file `vsmec_train.csv`, `vsmec_dev.csv`, `vsmec_test.csv` theo thứ tự cố định, ghép thành chuỗi `"train:<sha256>|dev:<sha256>|test:<sha256>"` rồi băm `sha256_str(...)`.
   - Giá trị hiện tại trên `data/clean/`:
     - `vsmec_data_hash` = `680bfc0f241ec6d4ef87e3e37c6d7552591779f40182b90c6654173c02960f95`
   - Vị trí lưu: cột `data_hash` trong `results/stage1.csv` và khóa `"data_hash"` trong từng file sidecar JSON `results/runs/<run_id>.json`.

2. **Bước 1.2 (`src/plm/train_neu3.py` — NEU-ESC 3 lớp):**
   - Hàm sử dụng: `sha256_file("data/clean/neuesc_100.csv")` trong `src/common/hashing.py`.
   - Giá trị hiện tại trên `data/clean/neuesc_100.csv`:
     - `neuesc_data_hash` = `9b0e3f42c2b7fa279e6887d15b522c566dd30d1c28430741972ed6d86ab03ac1`
   - Vị trí lưu: cột `data_hash` trong `results/neu3.csv` và khóa `"data_hash"` trong từng file sidecar JSON `results/neu3/runs/<run_id>.json`.

3. **Lưu ý về `data/clean/MANIFEST.json` (Kiểm tra F2):**
   - Tất cả **9 file dữ liệu cốt lõi** (`vsmec_{train,dev,test}.csv`, `neuesc_{100,20,5}.csv`, `vsfc_{100,20,5}.csv`, `overlap_excluded.csv`) đều **khớp 100%** với `MANIFEST.json`.
   - Tuy nhiên, 2 file `polarity_long.csv` và `polarity_nhat.csv` trong `MANIFEST.json` vẫn giữ mã băm lúc mới tạo template trống (`bba9a68c2af3...`), trong khi trên đĩa hai file này đã được điền nhãn ở CV-P3 (`054ffc6808...` và `a22f50bcd2...`), đồng thời `polarity_key.csv` chưa được thêm vào `MANIFEST.json`.
   - Tuân thủ tuyệt đối Nguyên tắc 1 ("Không sửa bất kỳ file nào trong `data/clean/`"), script `check_step2_data.py` giữ nguyên `MANIFEST.json` và báo cáo chính xác trạng thái `FAIL` ở mục `F2` để người hướng dẫn quyết định.

---

## 3. Kiểm chứng `src/data/check_overlap.py` (Mục 5.3)

Đọc trực tiếp `src/data/check_overlap.py` cho thấy script hiện tại được viết từ quy ước v2.x:
- Script chỉ tạo tập hash của `vsfc_test` và `neuesc_test` (`test_hashes`), sau đó loại các câu trùng khỏi `vsmec_train` (`exclude_from_vsmec_train`) và loại các câu trùng với `vsfc_test` khỏi `neuesc_train`.
- Script **chưa** kiểm tra chiều ngược lại theo chuẩn đối xứng của Bước 2 v3.0:
  - Chiều **D1**: câu trong `vsmec_test` xuất hiện ở bất kỳ split nào (`train`/`dev`/`test`) của `neuesc_100`.
  - Chiều **D2**: câu trong `neuesc_test` xuất hiện ở bất kỳ split nào (`train`/`dev`/`test`) của `vsmec`.

Khi chạy kiểm tra độc lập bằng `scripts/check_step2_data.py` trên dữ liệu thật:
- **D1 (exact match sau `normalize_text`):** **0 câu vi phạm (`PASS`)**.
- **D2 (exact match sau `normalize_text`):** **0 câu vi phạm (`PASS`)**.
- **D4 (near-duplicate sau khi bỏ toàn bộ dấu câu/emoji, giữ chữ cái tiếng Việt + chữ số):** phát hiện **2 cặp câu trùng phần chữ** (`WARN`, đã xuất ra `reports/step2/violations_D4_near_duplicates.csv`):
  1. `vsmec_test` (`id=6311`, label `1` - Sadness, text `"chạm đáy nỗi đau 😞"`) ↔ `neuesc_dev` (`id=25960`, label `0` - negative, text `"chạm đáy nỗi đau ."`).
  2. `neuesc_test` (`id=28482`, label `1` - neutral, text `"vật chất quyết định ý thức ."`) ↔ `vsmec_train` (`id=3607`, label `6` - Other, text `"vật chất quyết định ý thức"`).
  - Nguyên nhân: `neuesc_100.csv` đã qua bước tiền xử lý tách dấu câu ở cuối câu (thêm `" ."`), nên `normalize_text` (vốn giữ dấu chấm) sinh ra hai chuỗi khác nhau (`"chạm đáy nỗi đau 😞"` vs `"chạm đáy nỗi đau ."`), chỉ khi chuẩn hóa mạnh (bỏ dấu câu và emoji) mới lộ ra.

---

## 4. Hàm gộp xác suất `7 → 3` theo quy tắc B1 (Mục 5.4)

- Kiểm tra toàn bộ `src/` cho thấy hiện tại trong `src/common/constants.py` mới chỉ có ánh xạ nhãn cứng `HARD_MERGE_RULE: dict[int, int] = {0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}`, **chưa có** hàm gộp vector xác suất 7 chiều sang 3 chiều.
- Tuân thủ yêu cầu ở Mục 3.B2 ("Tuyệt đối không thêm hàm mới vào `src/` trong phạm vi công việc này"), hàm `aggregate_probs_7to3(p7)` đã được định nghĩa trực tiếp trong `scripts/check_step2_data.py` (và được import vào `tests/test_step2_data_ready.py` để kiểm thử các tính chất đại số ở nhóm B2).

---

## 5. Cách áp dụng `emoji_mode: keep` cho ViSoBERT (Mục 5.5)

Đối chiếu nhánh `visobert-vsmec-7c` (`src/plm/stage1.py`) và `Neu3-visobert` (`src/plm/train_neu3.py`):
- Khi huấn luyện ViSoBERT (`backbone: "uitnlp/visobert"`, `input_col: "text"`, `emoji_mode: "keep"`), dữ liệu đầu vào được biến đổi **in-memory** thông qua hàm `build_input_texts(df, "text", emoji_mode="keep")`:
  - Gọi `preprocess_text(str(t), emoji_mode="keep")` trong `src/common/textnorm.py` (chuẩn hóa Unicode NFC, lowercase, thu gọn ký tự lặp >2 xuống 2, chuẩn hóa khoảng trắng, giữ nguyên ký tự emoji gốc).
- Ngược lại, PhoBERT (`backbone: "vinai/phobert-base-v2"`, `input_col: "text_seg"`, `emoji_mode: None`) đọc trực tiếp chuỗi đã tách từ ở cột `text_seg` mà không gọi lại `preprocess_text`.
- Trong bài kiểm tra **I2–I4** của `scripts/check_step2_data.py`, cơ chế này được mô phỏng giống hệt lúc huấn luyện để đảm bảo độ dài token và đầu vào ở Bước 2 khớp tuyệt đối với Bước 1.1 / 1.2.

---

## 6. Câu hỏi cần xác nhận từ người hướng dẫn

Để chuyển trạng thái từ `CHƯA SẴN SÀNG` sang `SẴN SÀNG` trước khi chạy Bước 2, xin người hướng dẫn xác nhận các điểm sau:

1. **Tập `VSMEC-strict` (Mục E1–E3 — `BLOCKER`):**
   - Hiện tại các file `data/clean/polarity_{nhat,long,key}.csv` (100 câu: 50 `Surprise` + 50 `Other`, $\kappa = 0.4102$) được lấy mẫu từ **`vsmec_train.csv`** (phục vụ ước lượng ma trận `W_trans` ở CV-P3), **không phải** từ `vsmec_test.csv`.
   - Trong `vsmec_test.csv` hiện có **166 câu** thuộc hai lớp `Surprise` (37 câu) và `Other` (129 câu). Chúng ta sẽ gán nhãn độ phân cực cho 166 câu `Surprise`/`Other` của `vsmec_test.csv` để tạo `vsmec_strict` (kèm báo cáo $\kappa \ge 0.60$), hay sử dụng quy ước/file `vsmec_strict` nào khác?
2. **Đường dẫn 20 checkpoint và đồng bộ `results/stage1.csv`, `results/neu3.csv` (Mục F1, F3, G1, G3, J1–J3):**
   - Các lần chạy Bước 1.1 (`stage1.csv`) và Bước 1.2 (`neu3.csv`) cùng 20 thư mục checkpoint (`ckpt_roots` cho 4 tổ hợp × 5 seed) hiện nằm trên Google Drive / nhánh thực nghiệm. Xin xác nhận đường dẫn mount Drive (hoặc thư mục local) để điền vào `configs/step2_data.yaml`.
3. **Điểm chuẩn `vsmec_published_macro_f1` (Mục G2 — `BLOCKER`):**
   - Xin xác nhận con số `macro-F1` công bố trên VSMEC 7 lớp (ví dụ từ bài báo gốc Ho et al., 2020 hoặc chuẩn đối chiếu của nhóm) để điền vào `vsmec_published_macro_f1` trong `configs/step2_data.yaml`.
4. **Cập nhật `data/clean/MANIFEST.json` (Mục F2 — `BLOCKER`):**
   - Có cho phép cập nhật lại hash của `polarity_long.csv`, `polarity_nhat.csv` và bổ sung `polarity_key.csv` vào `data/clean/MANIFEST.json` (sau khi CV-P3 hoàn tất gán nhãn) hay không?
5. **Xử lý 2 cặp câu near-duplicate ở kiểm tra D4 (`WARN`):**
   - Cặp `vsmec_test id=6311` (`"chạm đáy nỗi đau 😞"`) ↔ `neuesc_dev id=25960` (`"chạm đáy nỗi đau ."`) và cặp `neuesc_test id=28482` (`"vật chất quyết định ý thức ."`) ↔ `vsmec_train id=3607` (`"vật chất quyết định ý thức"`): ở Bước 2 chúng ta giữ nguyên (vì tập dữ liệu đã đóng băng) và chỉ ghi chú trong báo cáo, hay loại khỏi tập đánh giá zero-shot?
