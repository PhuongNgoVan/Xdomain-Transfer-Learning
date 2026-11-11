# Quyết định kỹ thuật & Xác minh đầu vào Bước 1.2 (`Neu3-visobert`)

Tài liệu này ghi nhận kết quả xác minh nhánh `Neu3-visobert` theo **Mục 3** của `tasks/AGENT_neu3_visobert.md`, các số liệu thực tế tính từ `data/clean/neuesc_100.csv`, và các điểm chênh lệch cần người phụ trách xác nhận.

---

## 1. Kết quả kiểm tra danh mục Mục 3

### 1.1. Hỗ trợ `bf16` và `strict_precision`
- **Trạng thái ban đầu:** Nhánh `Neu3-visobert` tại thời điểm bắt đầu trỏ ở commit `b4c5bbb` (chưa có commit `a3c1240` từ `visobert-vsmec-7c`). Sau khi báo cáo và được người phụ trách cho phép merge `visobert-vsmec-7c` (`a3c1240`) vào `Neu3-visobert`, nhánh đã có đầy đủ hàm `resolve_precision(precision, device, strict_precision)` trong `src/plm/trainer_utils.py`, `src/plm/input_transform.py`, và luồng `torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)` trong `src/plm/stage1.py`.
- **Hành vi `bf16` trong `src/plm/train_neu3.py`:**
  - Tái sử dụng trực tiếp `resolve_precision(precision=precision, device=device, strict_precision=strict_precision)` từ `src.plm.trainer_utils`.
  - Khi `precision == "bf16"` và chạy trên GPU hỗ trợ `bfloat16`: `amp_dtype = torch.bfloat16`, `amp_enabled = True`, **không dùng `GradScaler`** (`use_fp16 = False`), thực hiện `clip_grad_norm_` trực tiếp trước `optimizer.step()` và `scheduler.step()`.
  - Khi `strict_precision: true` nhưng thiết bị không đáp ứng (ví dụ chạy CPU hoặc GPU không hỗ trợ `bfloat16`): `resolve_precision` lập tức bắn `RuntimeError`.
  - Log đầu mỗi lượt (`[NEU3 Precision]`) in rõ `requested`, `effective`, `autocast_dtype` (`torch.bfloat16` / `torch.float16` / `torch.float32 (autocast disabled)`), `strict_precision`, và tên GPU.

### 1.2. Cơ chế `emoji_mode: keep` cho ViSoBERT
- **Vị trí áp dụng:** Giống hệt Bước 1.1 (`visobert-vsmec-7c`), biến đổi văn bản được thực hiện **hoàn toàn trong bộ nhớ (in-memory) lúc nạp dữ liệu trước khi tokenize** thông qua `build_input_texts(df=df_split, input_col="text", alias="ViSoBERT", emoji_mode="keep")` từ `src/plm/input_transform.py`.
- **Chi tiết biến đổi:** Gọi `preprocess_text(str(t), emoji_mode="keep")` trên cột `text` của `data/clean/neuesc_100.csv` (chuẩn hóa Unicode NFC, đưa về chữ thường, thu gọn ký tự lặp, giữ nguyên Unicode emoji và ASCII emoticon, không sửa bất kỳ file nào trong `data/clean/`).
- **Ghi log & `variant`:**
  - Log đầu lượt in rõ `input_col=text | emoji_mode=keep` và dòng `[Tokenization] neuesc_{split} (text, emoji_mode=keep): ...`.
  - Sidecar `<results_dir>/runs/<run_id>.json` ghi `"emoji_mode": "keep"` và `"transformed_text_sha256"` của từng split đã nạp.
  - Cấu hình `configs/exp/neu3_visobert.yaml` đặt `emoji_mode: keep` và `variant: base-emokeep-bf16` (hậu tố `emokeep`, không chứa dấu `_`), sinh `run_id` dạng `ViSoBERT_NEU3-base-emokeep-bf16_s0_lr2e-5`.

### 1.3. Đối chiếu siêu tham số thực tế của Bước 1.1 (VSMEC)
Từ `configs/exp/stage1_visobert.yaml` và `src/plm/stage1.py` trên nhánh:

| Tham số | Giá trị thực tế Bước 1.1 (VSMEC) | Giá trị áp dụng cho Bước 1.2 (`neu3_visobert.yaml` / `train_neu3.py`) | Ghi chú |
| --- | --- | --- | --- |
| `batch_size` | `16` | `16` | Giữ nguyên |
| `gradient_accumulation_steps` (`grad_accum`) | `2` | `2` (batch hiệu dụng = 32) | Giữ nguyên |
| `num_train_epochs` (`epochs`) | `10` | `10` | Giữ nguyên |
| `warmup_ratio` | `0.06` | `0.06` | Giữ nguyên |
| `weight_decay` | `0.01` | `0.01` | Giữ nguyên |
| `max_grad_norm` | `1.0` | `1.0` | Giữ nguyên |
| `dropout` | `0.1` | `0.1` | Giữ nguyên |
| `max_length` | `256` | `256` (`dynamic_padding: true`, `group_by_length: true`) | Giữ nguyên |
| `eval_steps` | `max(1, steps_per_epoch // 2)` (tức 0.5 epoch/lần nếu không ghi đè trong YAML) | `max(1, steps_per_epoch // 2)` (với `n_train=23,048`, `steps_per_epoch=721` -> `eval_steps=360`) | Giữ nguyên công thức của `stage1.py` |
| `early_stopping_patience` | **`5`** | **`5`** | **Chênh lệch với Workflow v3.0 mục 0** (ghi patience = `2`). Theo quy tắc Mục 3, giữ nguyên giá trị `5` của VSMEC cho tới khi người phụ trách quyết định khác. |

### 1.4. Kiểm tra dữ liệu `data/clean/neuesc_100.csv` và `data/clean/MANIFEST.json`
- **Tồn tại & Schema:** `data/clean/neuesc_100.csv` có mặt đầy đủ với các cột `['id', 'text', 'label', 'subset', 'text_seg']` (ViSoBERT sử dụng cột `text`).
- **SHA-256 (`data_hash`):** `9b0e3f42c2b7fa279e6887d15b522c566dd30d1c28430741972ed6d86ab03ac1` — **khớp 100%** với khóa `"neuesc_100.csv"` trong `data/clean/MANIFEST.json`.
- **Cách `get_git_commit()` lấy commit:** Gọi `subprocess.check_output(["git", "rev-parse", "HEAD"])` trong `src/plm/trainer_utils.py`, trả về chuỗi SHA-1 đầy đủ 40 ký tự của commit `HEAD` (hoặc `"unknown"` nếu ngoài git repo).
- **Tính hợp lệ của văn bản & nhãn:**
  - `id`: 32,966 giá trị duy nhất, không trùng lặp, không NaN.
  - `text`: 0 dòng NaN, 0 dòng rỗng/chỉ chứa khoảng trắng.
  - `label`: chỉ gồm `{0, 1, 2}` (`0=negative`, `1=neutral`, `2=positive`).
- **Thống kê kích thước và phân bố nhãn thực tế (tính trực tiếp từ `data/clean/neuesc_100.csv`):**

| Subset | Tổng số mẫu | `0` (`negative`) | `1` (`neutral`) | `2` (`positive`) |
| --- | ---: | ---: | ---: | ---: |
| `train` | **23,048** | 4,215 (18.29%) | 15,936 (69.14%) | 2,897 (12.57%) |
| `dev` | **3,305** | 609 (18.43%) | 2,279 (68.96%) | 417 (12.62%) |
| `test` | **6,613** | 1,221 (18.46%) | 4,558 (68.92%) | 834 (12.61%) |
| **Tổng cộng** | **32,966** | **6,045 (18.34%)** | **22,773 (69.08%)** | **4,148 (12.58%)** |

### 1.5. Kiểm tra chồng lấp câu chéo tập VSMEC ↔ NEU-ESC
- Theo `data/clean/overlap_matrix.csv`, số câu trùng giữa `{vsmec_train, vsmec_dev, vsmec_test}` và `{neuesc_train, neuesc_dev, neuesc_test}` đều bằng **`0`**. Không có rò rỉ dữ liệu giữa VSMEC và tập test NEU-ESC.

---

## 2. Các điểm chênh lệch & Câu hỏi dành cho người phụ trách

1. **`early_stopping_patience`: `5` (code VSMEC Bước 1.1) so với `2` (Workflow v3.0 Mục 0)**
   - **Hiện tại:** Đang giữ `early_stopping_patience: 5` trong `configs/exp/neu3_visobert.yaml` đúng theo hướng dẫn tại Mục 3 (*"ghi chênh lệch vào `docs/neu3_decisions.md` và dùng giá trị của VSMEC cho tới khi người phụ trách trả lời"*).
   - **Câu hỏi:** Khi chạy 8 lượt thật trên Colab cho Bước 1.2, người phụ trách muốn giữ `early_stopping_patience: 5` (đồng nhất với Bước 1.1 VSMEC, tương đương 2.5 epoch khi `eval_steps = 0.5 epoch`) hay đổi sang `2`?

