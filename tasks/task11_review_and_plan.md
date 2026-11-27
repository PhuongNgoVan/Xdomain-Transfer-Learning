# Báo cáo Rà soát Toàn bộ Code Task 11 & Kế hoạch Chỉnh sửa (XLM-R & Open-Weight LLM)

- **Giao thức áp dụng:** `phase2-v1` ([`configs/phase2.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/phase2.yaml), [`reports/phase2/decisions.md`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/reports/phase2/decisions.md))
- **Tài liệu chuẩn đối chiếu:** [`AGENTS.md`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/AGENTS.md), [`docs/workflow_phase2_action_plan.md`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/docs/workflow_phase2_action_plan.md) (mục Task 11), [`notebooks/phase2_task08_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task08_colab.ipynb)
- **Trạng thái rà soát:** Hoàn tất rà soát tĩnh toàn bộ mã nguồn, cấu hình, kiểm thử và notebook Colab của Task 11. **Chưa chỉnh sửa mã nguồn hay chạy GPU trong bước này.**

---

## 1. Phạm vi rà soát & Đánh giá tổng quan

### 1.1. Các tệp đã rà soát
- **Cấu hình:**
  - [`configs/phase2.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/phase2.yaml) (hợp đồng `phase2-v1` đã khóa — bất biến).
  - [`configs/task11_xlmr_smoke.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/task11_xlmr_smoke.yaml), [`configs/task11_xlmr_sweep.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/task11_xlmr_sweep.yaml), [`configs/task11_llm.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/task11_llm.yaml).
- **Mã nguồn lõi & Task 11:**
  - [`src/phase2/llm.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py)
  - [`src/phase2/models.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/models.py)
  - [`src/phase2/select_lr.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/select_lr.py)
  - [`src/phase2/run.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/run.py), [`src/phase2/cli.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/cli.py), [`src/phase2/data.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/data.py), [`src/phase2/trainer.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/trainer.py), [`src/phase2/protocol.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/protocol.py)
  - [`src/zs/infer.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/zs/infer.py), [`src/zs/views.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/zs/views.py), [`src/zs/ckpt.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/zs/ckpt.py), [`src/plm/model_ckpt.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/plm/model_ckpt.py)
- **Notebook Colab & Đối chiếu:**
  - [`notebooks/phase2_task11_xlmr_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb)
  - [`notebooks/phase2_task11_llm_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_llm_colab.ipynb)
  - Mẫu chuẩn đối chiếu: [`notebooks/phase2_task08_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task08_colab.ipynb)
- **Kiểm thử & Quản lý mã nguồn:**
  - [`tests/test_phase2_task11_llm.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/tests/test_phase2_task11_llm.py), [`tests/test_phase2_task05.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/tests/test_phase2_task05.py), [`.gitignore`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/.gitignore)

### 1.2. Kết luận sơ bộ
Khung huấn luyện lõi của Phase 2 (`run.py`, `trainer.py`, `protocol.py`, `data.py`) đã đảm bảo tách biệt `train`/`dev`/`test`, `BF16` strict và `run_id` chống đụng độ. Tuy nhiên, phần mã riêng cho Task 11 và hai notebook Colab hiện có **6 lỗi nghiêm trọng (`P0`)**, **3 lỗ hổng tái lập & tổng hợp (`P1`)**, và **3 vấn đề tất định / kiểm thử (`P2`)**. Cần xử lý triệt để các lỗi này và commit cố định SHA trước khi chạy bất kỳ phiên GPU nào trên Google Colab.

| Mức độ | Số lượng | Tóm tắt trọng tâm |
|---|---:|---|
| **`P0` (Chặn chạy thật)** | **6** | Tràn SSD Colab (~156 GB ở XLM-R), ghi đè môi trường Colab + không khóa SHA, cắt cụt mất câu hỏi ở LLM few-shot, parser LLM bắt nhầm phủ định/đa nhãn, đếm sai ~50% tham số 4-bit & lệch benchmark chi phí LLM, thiếu khóa LR DFT 100% + kiểm tra nguồn XLM-R lỏng lẻo |
| **`P1` (Tái lập & Báo cáo)** | **3** | `llm.py` thiếu `completion.json`/`SKIP`/kiểm tra `model_meta`; thiếu pipeline `ZS` chuẩn, thiếu benchmark suy luận `ViSoBERT` trên cùng GPU & thiếu bộ bảng tổng hợp hiệu năng–chi phí; hai notebook thiếu kiến trúc vận hành 10 bước an toàn từ Task 08 |
| **`P2` (Tất định & Kiểm thử)** | **3** | Thứ tự cố định ở `stratified_fixed` & tie-break chưa khóa cứng trong `llm.py`, thiếu 2 góc nhìn `argmax` ở `vsmec`; thiếu toàn bộ test `XLM-R` + test biên `LLM` + `.gitignore` ẩn `test_phase2_*`; rủi ro lệch `source_code_sha256` nếu sửa `src/phase2/*.py` giữa chừng |

---

## 2. Chi tiết các lỗi & vấn đề phát hiện

### Nhóm `P0` — Lỗi nghiêm trọng bắt buộc sửa trước khi chạy Colab

#### `P0-1` — Tràn ổ cứng SSD `/content` (~156 GB) khi chạy XLM-R trên Colab
- **Vị trí:** [`notebooks/phase2_task11_xlmr_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb), [`src/phase2/trainer.py#L135-L145`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/trainer.py#L135-L145).
- **Nguyên nhân:** `xlm-roberta-base` có ~278M tham số (~1,11 GB `model.safetensors`, lớn gấp ~2,85 lần `ViSoBERT`). Ở mỗi lượt `Phase2Trainer.fit`, tệp `work/<run_id>/resume.pt` lưu `model` FP32 + `best_state` FP32 + trạng thái `AdamW` (`exp_avg` và `exp_avg_sq` FP32), chiếm khoảng **4,4 GB/lượt**.
- **Hệ quả:** Notebook XLM-R chạy `1 smoke + 12 sweep + 18 main = 31` lượt mà **không xóa `resume.pt`** sau khi hoàn tất (`complete=True`), đồng thời không truyền `--no-save-ckpt` cho 12 lượt `DFT 10%` và `Seq 10%` (`~20 GB` checkpoint không dùng lại). Tổng dung lượng SSD bị chiếm lên tới **~156 GB**, chắc chắn gây lỗi `Errno 28: No space left on device` giữa chừng trên Colab.

#### `P0-2` — Phá vỡ môi trường Python/CUDA của Colab & không khóa commit SHA ở cả 2 notebook
- **Vị trí:** [`notebooks/phase2_task11_xlmr_colab.ipynb#L39-L52`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb#L39-L52), [`notebooks/phase2_task11_llm_colab.ipynb#L37-L51`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_llm_colab.ipynb#L37-L51).
- **Nguyên nhân:**
  1. Cài trực tiếp `pip install -r requirements.txt` vào kernel Colab: ép cài `torch==2.9.0` đè lên PyTorch CUDA của Colab (profile chuẩn `colab-task06-v1` là `torch==2.14.1+cu130` / CUDA `13.0`), đồng thời cài `pandas==3.1.0` và `protobuf==7.36.2` vào môi trường hệ thống gây xung đột với `google-colab` (`requires pandas==2.2.3`).
  2. Dùng `git clone --no-checkout` (lỗi từng ghi nhận tại Task 06) kết hợp `git pull --ff-only` trên tên nhánh động `phase2_task11-xlmr-llm`, không cố định SHA 40 ký tự (`detached HEAD`).
- **Hệ quả:** Nếu nối lại phiên Colab sau khi nhánh có commit mới, [`src/phase2/select_lr.py#L41`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/select_lr.py#L41) sẽ báo `Mixed git_commit in sweep` và [`src/phase2/run.py#L133`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/run.py#L133) sẽ báo `Completion fingerprint mismatch`.

#### `P0-3` — Cắt cụt bên phải (`right-truncation`) làm mất câu cần phân loại ở LLM few-shot
- **Vị trí:** [`src/phase2/llm.py#L255-L261`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L255-L261), [`src/phase2/llm.py#L364-L368`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L364-L368), [`src/phase2/llm.py#L411-L417`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L411-L417), [`configs/task11_llm.yaml#L10`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/task11_llm.yaml#L10).
- **Nguyên nhân:** `configs/task11_llm.yaml` đặt `max_input_length: 1024`. Trong `HuggingFaceLLMClassifier.generate_batch`, `self.tokenizer(..., truncation=True, max_length=1024)` dùng mặc định `truncation_side="right"`. Trong khi đó, `build_llm_messages` đặt 6 (`neuesc`) hoặc 7 (`vsmec`) ví dụ minh họa ở **phần đầu**, còn `"Văn bản cần phân loại:\n{query_text}\nNhãn:"` và hậu tố chat template (`<|im_start|>assistant\n`) nằm ở **cuối cùng**.
- **Hệ quả:** Khi `rubric` + 7 bình luận dài vượt quá `1.024` token, phần câu cần phân loại và thẻ mở đầu câu trả lời bị cắt bỏ âm thầm mà không có kiểm tra hay ghi nhận `trunc_rate`.

#### `P0-4` — `parse_llm_label` bắt nhầm nhãn trong câu phủ định / đa nhãn và gán sai `valid_parse=True`
- **Vị trí:** [`src/phase2/llm.py#L274-L310`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L274-L310).
- **Nguyên nhân:** Nhánh fallback ở dòng `298-306` quét `\b{name}\b` trên toàn bộ văn bản sinh ra và lấy nhãn xuất hiện sớm nhất làm kết quả hợp lệ (`valid=True`).
- **Hệ quả:** Nếu mô hình sinh `"Không phải positive, câu này là negative"`, `"Trong [negative, neutral, positive], đáp án: positive"`, hoặc `"Sadness hoặc Anger"`, hàm vừa gán **sai nhãn** (lấy từ xuất hiện đầu tiên), vừa đánh dấu `valid_parse=True` (`invalid_output_rate = 0.0`), làm sai lệch tiêu chí phá hòa và chất lượng chọn prompt trên `dev`.

#### `P0-5` — Đếm sai ~50% tham số LLM ở chế độ `4bit` & sai lệch phép đo chi phí LLM
- **Vị trí:** [`src/phase2/llm.py#L397`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L397), [`src/phase2/llm.py#L428-L429`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L428-L429), [`src/phase2/llm.py#L763-L835`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L763-L835).
- **Nguyên nhân:**
  1. Ở chế độ lượng tử hóa `4bit` (`bitsandbytes` `Params4bit`), hai trọng số 4-bit được đóng gói vào 1 byte `uint8`, nên `sum(p.numel() for p in self.model.parameters())` tại dòng `397` trả về khoảng **một nửa** số tham số thực (~3,8B thay vì ~7,6B). Cần dùng `self.model.num_parameters()` (và ghi thêm `model_memory_footprint_mb`).
  2. `benchmark_llm_inference` tại dòng `777` chỉ tạo prompt `zero_shot` + `concise` (`examples=[]`), bỏ qua cấu hình `few_shot` đã khóa (vốn dài gấp 5–8 lần, tiêu tốn nhiều VRAM KV-cache và thời gian prefill hơn hẳn).
  3. Tại dòng `428-429`, `generate_batch` gọi `print(..., flush=True)` ở mỗi lần chạy với `batch_size=1`, cộng thêm chi phí I/O `stdout` vào độ trễ 50 lượt đo `latency_batch1`.

#### `P0-6` — Lỗ hổng khóa LR cho `DFT 100%` & xác minh checkpoint nguồn `XLM-R`
- **Vị trí:** [`notebooks/phase2_task11_xlmr_colab.ipynb#L98-L186`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb#L98-L186), [`src/phase2/models.py#L73-L99`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/models.py#L73-L99), [`src/phase2/models.py#L113-L119`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/models.py#L113-L119), [`src/phase2/select_lr.py#L18-L95`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/select_lr.py#L18-L95).
- **Nguyên nhân:**
  1. `select_phase2_p1_lr` hiện yêu cầu có đủ cả `dft` và `seq` mới xuất file khóa LR. Vì `Seq sweep` của XLM-R phụ thuộc vào checkpoint `DFT 100%`, Cell 2 của notebook XLM-R đã lọc thủ công `phase2_runs.csv` để chọn `dft_lr` (bỏ qua kiểm tra `fraction == 10`, `seed == 0`, `max_examples == ""`, `protocol_cfg`, hash artifact và cách ly `test`) rồi chạy Cell 3 (`DFT 100%`) **không truyền `--selection-path`**.
  2. `_locate_phase2_xlmr_source_checkpoint` (`models.py:73-99`) chỉ kiểm tra tồn tại `head.pt` + `encoder/` và không nằm dưới thư mục chứa chữ `sweep`/`smoke`. Nếu có một lượt chạy thử `XLMR_SA_dft_f100_s0_...` với `--run-mode main --max-examples 50`, hàm vẫn nạp nhầm làm checkpoint nguồn mà không kiểm tra `completion.json` (`complete == True`) hay `metadata.json` (`run_mode == "main"`, `max_examples is None`, `protocol_version == "phase2-v1"`).
  3. `resolve_model_source` (`models.py:119`) gán cứng `"historical_run_provenance": "unverified"` cho cả `XLMR`, dù checkpoint nguồn `XLMR` được huấn luyện trực tiếp trong Giai đoạn 2 theo `phase2-v1`.

---

### Nhóm `P1` — Lỗ hổng khả năng phục hồi, đo chi phí & tổng hợp kết quả

#### `P1-1` — `src/phase2/llm.py` thiếu `completion.json`, kiểm tra `model_meta` & cơ chế `SKIP`
- **Vị trí:** [`src/phase2/llm.py#L527-L761`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L527-L761).
- **Chi tiết:**
  - `dev-sweep` trên `neuesc` chạy `8 tổ hợp × 3.305 câu = 26.440` lượt sinh; nếu Colab ngắt kết nối ở tổ hợp thứ 7/8, toàn bộ 6 tổ hợp trước bị chạy lại từ đầu vì thiếu `completion_<tag>.json` + kiểm tra hash để `SKIP`.
  - `run_llm_test_eval` chưa xác minh `model_name` và `precision` hiện tại có khớp `selection_llm_{domain}.json` hay không, và chưa xuất `completion.json` chứa SHA-256 các artifact đầu ra theo quy ước [`AGENTS.md`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/AGENTS.md).

#### `P1-2` — `ZS`, benchmark chi phí suy luận và tổng hợp bảng Task 11 rời rạc, thiếu đối sánh chuẩn với `ViSoBERT`
- **Vị trí:** [`notebooks/phase2_task11_xlmr_colab.ipynb#L274-L424`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb#L274-L424).
- **Chi tiết:**
  - **Suy luận `ZS`:** Cell 7 tự viết vòng lặp trong kernel, không tắt `autocast`, không sắp xếp theo độ dài câu như [`predict_checkpoint`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/zs/infer.py#L64-L84), không kiểm tra `check_underscore_leakage`, giữ mô hình trên VRAM kernel và không tạo `completion.json`/`SKIP`.
  - **Đo chi phí suy luận không được lưu & thiếu `ViSoBERT`:** Cell 8 chỉ `print()` độ trễ `batch=1` và throughput `batch=32` của `XLMR` trên `SA` ra màn hình mà **không lưu ra file JSON** (nên Cell 9 đóng gói mất hoàn toàn số đo tốc độ suy luận). Quan trọng hơn, [`docs/workflow_phase2_action_plan.md#L212`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/docs/workflow_phase2_action_plan.md#L212) yêu cầu đo **trên cùng môi trường GPU chuẩn** giữa `ViSoBERT`, `XLM-R` và `LLM`; Task 08 mới chỉ có `train_seconds` và `peak_vram_mb` khi train, chưa có `latency batch=1`, `throughput batch=32` và `GPU-h/1.000 câu` khi suy luận cho `ViSoBERT`.
  - **Thiếu tổng hợp kết quả Task 11:** Chưa có module tổng hợp các bảng `metrics_summary`, `per_class_f1` (tuân thủ `V3-VSMEC-strict` **chỉ xuất `negative` và `positive`**), `exploratory_contrasts` (`Seq − DFT`, `Seq − ZS` trên cùng view 3 lớp, `XLMR − ViSoBERT` ở 10% trên 3 seed chung `0..2`, tất cả gắn `exploratory=True` để không chạm `Holm m=2`), và bảng đánh đổi hiệu năng–chi phí hợp nhất.

#### `P1-3` — Thiếu kiến trúc vận hành Colab an toàn học từ Task 08 ở cả 2 notebook
- **Vị trí:** [`notebooks/phase2_task11_xlmr_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb), [`notebooks/phase2_task11_llm_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_llm_colab.ipynb).
- **Chi tiết:**
  - Chạy `--max-examples` (smoke) vào cùng `RESULTS_DIR` thay vì thư mục tách biệt `results_smoke` (dễ làm bẩn `phase2_runs.csv` và cản trở `select_lr.py:39`).
  - Không đồng bộ nguyên tử lên Google Drive phẳng và không có `hydrate_from_flat_drive` khi nối phiên; tải file qua `google.colab.files.download` dễ lỗi khi mất kết nối trình duyệt.
  - Không chụp `frozen_before.json`, không dùng subprocess kèm heartbeat 30s và xử lý tín hiệu `SIGINT`/`SIGTERM`, không có bước xác minh `SKIP` và kiểm định độc lập đầu ra.

---

### Nhóm `P2` — Tính tất định, tương thích schema & khoảng trống kiểm thử

#### `P2-1` — Tính tất định khi phá hòa & đồng bộ schema góc nhìn nhãn ở `src/phase2/llm.py`
- **Vị trí:** [`src/phase2/llm.py#L176-L226`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L176-L226), [`src/phase2/llm.py#L493-L497`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L493-L497), [`src/phase2/llm.py#L627-L641`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/llm.py#L627-L641).
- **Chi tiết:**
  - `stratified_fixed` luôn giữ nguyên thứ tự lớp `0..K-1`, khiến ví dụ cuối cùng ngay sát câu hỏi luôn thuộc lớp `Other` (`vsmec`) hoặc `positive` (`neuesc`), dễ tạo thiên lệch vị trí cuối (*recency bias*); nên xáo trộn thứ tự các ví dụ cố định một cách tất định bằng `rng` từ `seed`.
  - `np.argsort` trong `tfidf_knn` và `balanced_tfidf_knn` dùng `quicksort` mặc định (không ổn định khi điểm bằng nhau); nên dùng `kind="mergesort"` để tất định tuyệt đối.
  - `run_llm_dev_sweep` cần thêm tiêu chí phá hòa thứ cấp tất định (`strategy`, `prompt_style`) sau `dev_selection_score` và `invalid_output_rate`.
  - `evaluate_llm_split` trên `vsmec` mới xuất 4 view; nên xuất đủ 6 view (bổ sung `V3-VSMEC/argmax` và `V3-VSMEC-strict/argmax`) để khớp 1-1 với `required_views("SB", "main")` trong [`src/phase2/run.py#L57-L65`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/run.py#L57-L65).

#### `P2-2` — Khoảng trống Unit Test & cấu hình `.gitignore`
- **Vị trí:** [`tests/test_phase2_task11_llm.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/tests/test_phase2_task11_llm.py), [`.gitignore#L2-L6`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/.gitignore#L2-L6).
- **Chi tiết:**
  - Chưa có tệp test cho nhánh `XLM-R` (`tests/test_phase2_task11_xlmr.py`) và cho module tổng hợp Task 11.
  - `test_phase2_task11_llm.py` chưa kiểm thử: bảo vệ không cắt cụt câu hỏi đích, parser với câu phủ định/đa nhãn, `SKIP`/hỏng hash ở `dev-sweep` & `eval-test`, từ chối lệch `model_name`/`precision`, đếm tham số mock `4bit` và `benchmark_llm_inference`.
  - `.gitignore` dòng 2 (`tests/*`) đang bỏ qua toàn bộ `tests/test_phase2_*.py`; nếu không whitelist `!tests/test_phase2_*.py` khi commit lên nhánh, bản clone trên Colab sẽ không có các file test mới này.

#### `P2-3` — Tác động của `source_code_sha256` trong `src/phase2/run.py`
- **Vị trí:** [`src/phase2/run.py#L116`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/src/phase2/run.py#L116).
- **Chi tiết:** `run.py` đưa `sha256` của mọi file `src/phase2/*.py` vào `fingerprint` của mỗi run. Điều này **không** làm hỏng `selection_p1_lr.json` của Task 07 (vì file JSON chọn LR chỉ giữ `protocol_hash`, `data_hash`, `ids_sha256`), nhưng **trong phạm vi Task 11**, toàn bộ chỉnh sửa trong `src/phase2/*.py` phải được hoàn thiện, kiểm thử và commit cố định SHA **trước** khi chạy bất kỳ lượt `sweep`/`main` nào trên Colab; không sửa code `src/phase2/*.py` sau khi đã chạy `DFT 100%` hoặc `sweep` nếu muốn `SKIP` và `select_lr` thông qua kiểm tra đồng nhất mã nguồn.

---

## 3. Bài học từ `notebooks/phase2_task08_colab.ipynb` & Kiến trúc áp dụng cho Task 11

### 3.1. Đối chiếu mô hình 10 bước chuẩn của Task 08 với Task 11

| Bước | Chuẩn trong `phase2_task08_colab.ipynb` | Thực tế hiện tại ở 2 notebook Task 11 | Thiết kế mục tiêu sau chỉnh sửa |
|---|---|---|---|
| **Bước 1** | Khai báo tập trung `SESSION_ID`, `REPO_REF` (hỗ trợ SHA 40 ký tự), `PROFILE_ID`, đường dẫn Drive phẳng, chính sách giữ checkpoint & xóa `resume.pt` | Khai báo rời rạc vài biến chuỗi, thiếu `PROFILE_ID`, không có đường dẫn Drive phẳng hay cờ dọn `resume.pt` | Áp dụng cấu hình tập trung tương tự Task 08 cho `Task11_XLMR` (`colab-task06-v1`) và `Task11_LLM` (`colab-task11-llm-v1`) |
| **Bước 2** | Kiểm tra GPU/BF16 & phiên bản Torch/CUDA trước khi mount Drive; giữ kernel 0 MiB VRAM | Không khớp profile Torch/CUDA; notebook XLM-R chỉ in sơ sài 3 dòng; LLM tự pip install trong cell sau | Kiểm tra GPU/BF16, khớp profile Torch/CUDA trước khi mount Drive; tuyệt đối không giữ model trên kernel |
| **Bước 3** | Clone an toàn (tránh `--no-checkout`), checkout `detached HEAD`, khóa `COMMIT` 40 ký tự, chụp `frozen_before.json`, `hydrate_from_flat_drive` | Dùng `--no-checkout` + `git pull --ff-only`, không `frozen_before.json`, không `hydrate` | Dùng hàm git + `frozen_before.json` + `hydrate_from_flat_drive` kiểm tra khớp `commit`, `profile_id` và SHA-256 artifact |
| **Bước 4** | Kiểm tra `venv`/`ensurepip`, tự cài `pythonX.Y-venv` nếu thiếu, dọn venv hỏng | Không tạo virtualenv riêng | Dùng virtualenv riêng trên `/content` theo đúng profile |
| **Bước 5** | Tạo `venv --system-site-packages`, cài đúng các gói pin, ghi `install.log`, `pip-freeze.txt`, `pip_check.txt`, cấu hình `RUN_ENV` (`USE_TF=0`, `HF_TOKEN`) | Chạy `pip install -r requirements.txt` thẳng vào kernel Colab, ghi đè `torch`/`pandas` | Cài đúng pin vào venv; giữ nguyên PyTorch CUDA hệ thống; ghi đầy đủ log cài đặt |
| **Bước 6** | Preflight trong subprocess venv: ghi `environment.json` **trước `assert`**, thử forward/loss BF16 (và verify license/revision đối với LLM) | Không có `environment.json`, không kiểm tra pin hay forward preflight | Subprocess preflight ghi `environment_<SESSION_ID>.json` + `environment.json`; kiểm tra BF16 forward (XLM-R) hoặc metadata + license HF (LLM) |
| **Bước 7** | Stage đầu vào & kiểm tra dữ liệu + `smoke` vào thư mục riêng | Chạy smoke chung vào `RESULTS_DIR`, không kiểm tra dữ liệu đầy đủ trước khi chạy hàng loạt | Chạy smoke vào thư mục `results_smoke` riêng biệt, kiểm tra đầy đủ schema/rò rỉ trước khi chạy sweep/main |
| **Bước 8** | Chạy từng lượt bằng `subprocess.Popen` trong `start_new_session=True`, heartbeat 30s, bắt `SIGINT`/`SIGTERM`, xóa `resume.pt` ngay khi xong, đồng bộ nguyên tử lên Drive phẳng | Gọi `subprocess.run` trơ hoặc chạy thẳng trong kernel, không dọn `resume.pt`, không backup Drive | Subprocess runner kèm heartbeat 30s, dọn `work/<run_id>/resume.pt` (~4,4 GB/lượt với XLM-R), chỉ lưu checkpoint `DFT 100%` (tắt ở `10%`), đồng bộ ngay sau mỗi lượt |
| **Bước 9** | Kiểm tra `SKIP` bằng CLI + script nghiệm thu độc lập toàn bộ `completion.json`, `provenance.json`, `predictions_test.csv`, `frozen_before.json` | Hoàn toàn không có | Chạy kiểm tra `SKIP` + nghiệm thu độc lập 100% artifact, `in_strict=526`, xác suất tổng 1 và `frozen_before.json` |
| **Bước 10** | Sinh toàn bộ bảng CSV & báo cáo `.md`, đồng bộ + kiểm chứng SHA-256 trên Drive phẳng | Chỉ `print()` sơ sài và tải `.tar.gz` qua trình duyệt | Sinh đầy đủ bảng CSV (`task11_*.csv`), báo cáo `.md`, đo chi phí suy luận chuẩn cả `ViSoBERT`–`XLM-R`–`LLM` và xác minh SHA-256 trên Drive |

### 3.2. Chiến lược tổ chức 2 notebook & chia sẻ tài nguyên
- **Giữ 2 notebook tách biệt** ([`phase2_task11_xlmr_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_xlmr_colab.ipynb) và [`phase2_task11_llm_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task11_llm_colab.ipynb)), đồng bộ cùng chuẩn 10 bước của Task 08 để chạy độc lập và phục hồi an toàn:
  - **Notebook XLM-R** dùng đúng profile `colab-task06-v1` (12 package pin y hệt Task 06/07/08) và sao lưu lên `/content/drive/MyDrive/Xdomain/Phase2/Task11_XLMR`. Đồng thời đo benchmark suy luận cho **cả `ViSoBERT` và `XLM-R`** trong cùng phiên GPU (ví dụ A100 hoặc L4) để bảng đánh đổi hiệu năng–chi phí cùng chuẩn phần cứng.
  - **Notebook LLM** dùng profile `colab-task11-llm-v1` (kế thừa 12 pin của `colab-task06-v1` + pin tường minh `accelerate` và `bitsandbytes` khi VRAM < 22 GiB), sao lưu lên `/content/drive/MyDrive/Xdomain/Phase2/Task11_LLM`.
- **Tổng hợp liên kết:** Module tổng hợp mới (`src/phase2/task11.py`) cho phép đọc từ thư mục kết quả hợp nhất (hoặc chỉ định riêng đường dẫn `Task08` ViSoBERT + `Task11_XLMR` + `Task11_LLM`) để xuất bảng so sánh thống nhất.

---

## 4. Kế hoạch chỉnh sửa chi tiết theo 7 Pha

### Pha 1 — Gia cố `src/phase2/select_lr.py`, `src/phase2/models.py` và `src/phase2/run.py` cho XLM-R

1. **`src/phase2/select_lr.py`:**
   - Mở rộng chữ ký: `select_phase2_p1_lr(results_dir, out=None, config_path="configs/phase2.yaml", backbone="ViSoBERT", methods=("dft", "seq"))` và thêm tùy chọn `--methods` (`"dft"`, `"dft,seq"`) ở CLI `main()`.
   - Mặc định `methods=("dft", "seq")` giữ nguyên 100% hành vi Task 07 cho `ViSoBERT`.
   - Khi `backbone="XLMR"` và `methods=("dft",)`, hàm kiểm tra đầy đủ 6 lượt `DFT sweep` (`fraction=10, seed=0, max_examples="", split="dev"`, đủ hash artifact, `protocol_cfg`, cảnh báo `grid_edge`), ghi `selection_xlmr_dft_lr.json` (và `xlmr_dft_lr_selection.csv`) để **cell `DFT 100%` bắt buộc truyền `--selection-path`**.
   - Sau 6 lượt `Seq sweep`, gọi lại với `methods=("dft", "seq")` để ghi `selection_xlmr_lr.json` (và kiểm tra `selected["XLMR/<dir>/dft"]` khớp bản đã khóa trước đó).
2. **`src/phase2/models.py`:**
   - Gia cố `_locate_phase2_xlmr_source_checkpoint(root, direction, seed)`:
     - Bên cạnh kiểm tra `head.pt` và `encoder/`, kiểm tra nếu có `results_dir` hoặc `metadata.json`/`completion.json` đi kèm trong thư mục cha/liền kề (hoặc đọc manifest `xlmr_source_manifest.json` được ghi khi `save_ckpt`), xác minh `complete=True`, `backbone="XLMR"`, `direction=source_dir`, `method="dft"`, `fraction=100`, `seed=seed`, `run_mode="main"`, `max_examples is None`, `protocol_version="phase2-v1"`.
     - Giải pháp sạch và không phụ thuộc đường dẫn tuyệt đối: khi `run.py` lưu checkpoint `main` (`save_ckpt=True`) cho `XLMR`, ghi thêm file xác nhận nhỏ hoặc kiểm tra tương ứng đối với các tệp trong thư mục kết quả/checkpoint; trong `_locate_phase2_xlmr_source_checkpoint(..., results_dir=None)` hỗ trợ xác minh chéo `completion.json` & `metadata.json` khi truyền `results_dir` (và từ chối ngay nếu tên/metadata cho thấy là `max_examples` bị cắt).
   - Cập nhật `resolve_model_source`: tách provenance nguồn `XLMR` (`"source_run_origin": "phase2_xlmr_dft_100pct"`) thay vì gán `"historical_run_provenance": "unverified"`, **giữ nguyên hoàn toàn nhánh `ViSoBERT` không đổi**.
3. **`src/phase2/run.py`:**
   - Nếu `selection_path` được truyền cho `XLMR` ở `DFT 100%`, đảm bảo `run.py` kiểm tra đúng khóa `XLMR/{direction}/dft` từ `selection_xlmr_dft_lr.json` hoặc `selection_xlmr_lr.json` (hiện tại `run.py:104-110` đã đọc đúng khóa `{backbone}/{direction}/{selected_method}` và khớp `data_hash` vì `data_hash` băm file nguồn `neuesc_100.csv` / `vsmec_*`, nên hoàn toàn tương thích giữa `10%` và `100%` — cần viết test xác minh tường minh).

### Pha 2 — Tạo module `src/phase2/task11.py` chuẩn hóa `ZS`, đo chi phí Encoder và tổng hợp Task 11

Tạo mới `src/phase2/task11.py` (không sửa bất kỳ file nào trong `src/zs/`, `src/seq/`, `src/stats/`) với 3 nhóm chức năng chính + CLI:
1. **`run_xlmr_zs_eval(results_dir, ckpt_dir, *, seeds=(0, 1, 2), clean_dir="data/clean", max_examples=None, batch_size=32)`:**
   - Chạy suy luận `ZS` hai chiều (`SA` dùng checkpoint nguồn `SB` `100%` 7 nhãn đánh giá `neuesc` qua `zsA`; `SB` dùng checkpoint nguồn `SA` `100%` 3 nhãn đánh giá `vsmec` qua `zsB`) trên `seed 0..2`.
   - Tái sử dụng `predict_checkpoint`, `_prediction_frame`, `score_prediction_view`, `validate_probabilities` từ `src/zs/infer.py` và `src/zs/mapping.py`; bắt buộc kiểm tra `check_underscore_leakage` với `emoji_mode="keep"`.
   - Xuất cấu trúc thư mục chuẩn `results/phase2/xlmr_zs/runs/<run_id>/`: `predictions_test.csv`, `metrics.json`, `metadata.json`, `cost.json`, `completion.json` (hỗ trợ `SKIP` theo fingerprint + SHA-256 artifact) và bảng tổng hợp `xlmr_zs_runs.csv`.
2. **`benchmark_encoder_inference(backbones=("ViSoBERT", "XLMR"), domains=("neuesc", "vsmec"), *, precision="fp32", clean_dir="data/clean", n_b1=100, large_batch_size=32, max_length=256, local_files_only=False, model_paths=None, out_path=None)`:**
   - Đo cùng điều kiện trên cùng GPU: `total_params`, `trainable_params`, `peak_vram_mb` (sau `torch.cuda.reset_peak_memory_stats()`), `latency_batch1_mean_ms` / `p50` / `p95` (sau warmup + `torch.cuda.synchronize()`), `throughput_samples_per_sec` (`batch_size=32`), và `gpu_hours_per_1000_samples`.
   - Lưu JSON đầy đủ kèm `gpu_name`, `torch`, `cuda`, `precision`, `batch_size`, `max_length`.
3. **`build_task11_tables_and_report(xlmr_results_dir, llm_results_dir, *, visobert_results_dir=None, encoder_cost_path=None, llm_cost_dir=None, out_dir="results/phase2/tables", report_path="reports/phase2/task11_xlmr_llm_10pct.md")`:**
   - Sinh 5 bảng CSV chuẩn hóa:
     1. `task11_xlmr_manifest.csv` (30 lượt train `XLM-R` + 6 lượt `ZS`).
     2. `task11_metrics_summary.csv` (`mean ± SD` trên mọi view của `ViSoBERT` 10% [nếu có], `XLM-R` `DFT 100%` / `DFT 10%` / `Seq 10%` / `ZS`, và `LLM` `zero_shot` / `few_shot`).
     3. `task11_per_class_f1.csv` (tuân thủ nghiêm ngặt quy tắc `AGENTS.md`: `V3-VSMEC-strict` **chỉ xuất `negative` và `positive`**).
     4. `task11_exploratory_contrasts.csv` (tính bằng `paired_gain` với `boot=10000, boot_seed=0`: `XLMR Seq − DFT`, `XLMR Seq − ZS` trên các view 3 lớp cùng hướng, và `XLMR − ViSoBERT` ở 10% trên 3 seed chung `[0, 1, 2]`; mọi dòng đều gắn `primary=False, exploratory=True`, tuyệt đối không động vào `Holm m=2`).
     5. `task11_cost_performance_10pct.csv` (so sánh `ViSoBERT`, `XLM-R`, `Qwen2.5-7B-Instruct`: `total_params`, `trainable_params`, `train_gpu_hours`, `peak_vram_mb`, `latency_batch1_ms`, `throughput_samples_per_sec`, `infer_gpu_hours_per_1k`, `Macro-F1` ở `SA` & `SB`).
   - Sinh báo cáo Markdown `reports/phase2/task11_xlmr_llm_10pct.md` ghi rõ ghi chú `pretraining_leakage_risk` cho LLM và điều kiện phần cứng khi đo.

### Pha 3 — Chỉnh sửa `src/phase2/llm.py` và `configs/task11_llm.yaml`

1. **Chống mất câu hỏi khi tokenize (`configs/task11_llm.yaml` & `src/phase2/llm.py`):**
   - Nâng `max_input_length: 2048` trong [`configs/task11_llm.yaml`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/configs/task11_llm.yaml).
   - Viết hàm phụ trợ kiểm soát độ dài prompt trước khi gọi `predictor_fn`: giữ nguyên 100% câu truy vấn đích + phần chỉ dẫn hệ thống + hậu tố chat template; chỉ rút ngắn phần văn bản của các ví dụ few-shot (ví dụ giới hạn tối đa ~160–256 token hoặc ~400 ký tự mỗi ví dụ) nếu tổng token vượt `max_input_length`, và **ném `ValueError` nếu câu truy vấn + system/suffix tự thân vượt trần** — tuyệt đối không để `right-truncation` cắt mất câu cần phân loại.
   - Ghi thêm `max_prompt_tokens`, `mean_prompt_tokens`, `truncated_example_count` vào kết quả `evaluate_llm_split` và `metrics.json`.
2. **Siết chặt `parse_llm_label`:**
   - Giữ ưu tiên khớp chính xác dòng đầu (đã bỏ dấu câu bao quanh), chuỗi JSON `{"label": ...}` và mẫu tường minh `Nhãn: <label>` / `Label: <label>`.
   - Ở nhánh quét từ biên cuối cùng:
     - Thu thập **tất cả** nhãn hợp lệ xuất hiện trong văn bản đầu ra.
     - Nếu có **từ 2 nhãn khác nhau trở lên** (ví dụ `"Không phải positive mà là negative"`, `"Sadness hoặc Anger"`, `"Trong [negative, neutral, positive]..."`) hoặc có từ phủ định đứng ngay trước nhãn đơn (`không phải`, `not`, `chứ không`), trả về `(fallback_id, fallback_name, False)` thay vì đoán bừa và gắn `valid=True`.
     - Chỉ trả `valid=True` khi tồn tại **duy nhất 1 nhãn hợp lệ** không bị phủ định.
3. **Sửa đếm tham số `4bit` & chuẩn hóa `benchmark_llm_inference`:**
   - Trong `HuggingFaceLLMClassifier.__init__`: dùng `int(self.model.num_parameters())` (fallback về `sum(p.numel() ...)` chỉ cho mô hình giả không có `num_parameters`) và bổ sung `model_memory_footprint_mb` (từ `get_memory_footprint()` nếu có).
   - Thêm tham số `quiet: bool = False` cho `generate_batch` để tắt `print(..., flush=True)` trong lúc benchmark.
   - Mở rộng `benchmark_llm_inference`: nhận `selection_path` (hoặc `examples_mode=("zero_shot", "few_shot")`), đo tách biệt hiệu năng trên cả `zero_shot` lẫn `few_shot` đã khóa ở chế độ `quiet=True`, ghi đầy đủ `peak_vram_mb`, `latency_batch1_*`, `throughput_samples_per_sec`, `gpu_hours_per_1000_samples` cho từng chế độ.
4. **Thêm `completion.json`, xác minh `model_meta` và cơ chế `SKIP`:**
   - Trong `run_llm_dev_sweep`: mỗi tổ hợp `(strategy, prompt_style)` ghi `dev_completion_<tag>.json` chứa fingerprint (`protocol_hash`, `data_hash`, `ids_sha256`, `model_meta`, `max_examples`, `shots_per_class`, `seed`, `source_code_sha256`) và SHA-256 của `dev_pred_<tag>.csv`, `dev_raw_<tag>.csv`. Nếu tổ hợp đã hoàn tất và khớp SHA-256 + fingerprint thì **SKIP** ngay; nếu hỏng hoặc thiếu file thì chạy lại.
   - Trong `run_llm_test_eval`: kiểm tra `model_meta` (`model_name` và `precision`) khớp với `selection_llm_{domain}.json` (nếu truyền `model_meta`), ghi `completion.json` cho từng `test_<mode_key>` (và kiểm tra `SKIP` an toàn khi đủ artifact + SHA-256 khớp).
5. **Tính tất định & đủ góc nhìn nhãn:**
   - `FewShotSelector`: xáo trộn tất định `self._fixed_indices` bằng `rng` (`seed`) để không dồn lớp cuối lên sát câu truy vấn; dùng `np.argsort(..., kind="mergesort")` trong `tfidf_knn` và `balanced_tfidf_knn`.
   - `run_llm_dev_sweep`: phá hòa thứ cấp tất định theo `("dev_selection_score", "invalid_output_rate", "strategy", "prompt_style")`.
   - `evaluate_llm_split` trên `vsmec`: bổ sung `("V3-VSMEC", "argmax")` và `("V3-VSMEC-strict", "argmax")` để đồng bộ 6 góc nhìn với `SB/main`.

### Pha 4 — Thiết kế lại `notebooks/phase2_task11_xlmr_colab.ipynb` (Chuẩn 10 bước Task 08)

Thiết kế lại toàn bộ notebook theo đúng 10 bước của [`phase2_task08_colab.ipynb`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/notebooks/phase2_task08_colab.ipynb):
- **Bước 1:** Khai báo `SESSION_ID`, `REPO_REF` (hỗ trợ SHA 40 ký tự), `PROFILE_ID = "colab-task06-v1"`, `DRIVE_BACKUP_BASE = Path("/content/drive/MyDrive/Xdomain/Phase2/Task11_XLMR")`, `CLEAN_COMPLETED_RESUME_PT = True`, `BACKUP_DFT100_CKPTS_TO_DRIVE = True`.
- **Bước 2:** Kiểm tra GPU CUDA/BF16, khớp `torch` + `cuda` với profile `colab-task06-v1`, mount Google Drive.
- **Bước 3:** Clone/fetch repo an toàn (không `--no-checkout`), checkout `detached HEAD`, khóa `COMMIT`, chụp `frozen_before.json`, chạy `hydrate_from_flat_drive` kiểm tra `commit` & `profile_id` và khôi phục cả artifact lẫn checkpoint `DFT 100%` từ Drive phẳng.
- **Bước 4–6:** Kiểm tra `venv`/`ensurepip`, tạo virtualenv `--system-site-packages`, cài đúng 12 pin, ghi log/freeze, thiết lập `RUN_ENV`, chạy preflight trong subprocess venv ghi `environment.json` và kiểm tra forward/loss BF16 cho `xlm-roberta-base`.
- **Bước 7:** Chạy smoke `configs/task11_xlmr_smoke.yaml --max-examples 50` vào thư mục riêng `/content/task11_xlmr_smoke` (tách ly hoàn toàn khỏi `RESULTS_DIR`), kiểm tra `predictions_test.csv` **không tồn tại**.
- **Bước 8:** Thực thi tuần tự các giai đoạn phụ thuộc qua `subprocess.Popen` (`start_new_session=True`), heartbeat 30s, xóa ngay `work/<run_id>/resume.pt` (~4,4 GB/lượt) và đồng bộ nguyên tử lên Drive sau mỗi lượt:
  1. `6` lượt `DFT sweep` (`10%`, `seed 0`, dev-only) → chạy `select_lr --backbone XLMR --methods dft` khóa `selection_xlmr_dft_lr.json`.
  2. `6` lượt `DFT 100%` (`seed 0..2`, truyền `--selection-path selection_xlmr_dft_lr.json`, **có lưu checkpoint**).
  3. `6` lượt `Seq sweep` (`10%`, `seed 0`, dev-only, nạp checkpoint `DFT 100%`) → chạy `select_lr --backbone XLMR --methods dft,seq` khóa `selection_xlmr_lr.json`.
  4. `6` lượt `DFT 10%` (`seed 0..2`, `--selection-path`, `--no-save-ckpt`).
  5. `6` lượt `Seq 10%` (`seed 0..2`, `--selection-path`, `--source-ckpt-root`, `--no-save-ckpt`).
  6. `6` lượt `ZS` (`SA/SB × seed 0..2`) qua `python -m src.phase2.task11 xlmr-zs`.
- **Bước 9:** Gọi lại CLI xác nhận `SKIP` cho `DFT 100%`, `DFT 10%`, `Seq 10%`, `ZS`, và chạy script nghiệm thu độc lập (30 lượt train + 6 lượt ZS, `in_strict=526`, tổng xác suất = 1, `frozen_before.json`).
- **Bước 10:** Đo benchmark suy luận chuẩn hóa cho **cả `ViSoBERT` và `XLMR`** trên cùng GPU qua `python -m src.phase2.task11 benchmark-encoders`, xuất các bảng CSV & báo cáo XLM-R, đồng bộ & kiểm chứng SHA-256 trên Drive phẳng.

### Pha 5 — Thiết kế lại `notebooks/phase2_task11_llm_colab.ipynb` (Chuẩn 10 bước Task 08)

Thiết kế lại toàn bộ notebook theo đúng 10 bước của Task 08:
- **Bước 1–2:** Khai báo `PROFILE_ID = "colab-task11-llm-v1"` (kế thừa pin của `colab-task06-v1` + pin tường minh `accelerate`, và thêm `bitsandbytes` khi VRAM < 22 GiB), `MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"`, `DRIVE_BACKUP_BASE = Path("/content/drive/MyDrive/Xdomain/Phase2/Task11_LLM")`, kiểm tra GPU/VRAM/BF16, mount Drive.
- **Bước 3–5:** Clone/fetch cố định `COMMIT`, chụp `frozen_before.json`, `hydrate_from_flat_drive`, tạo virtualenv độc lập, cài gói pin + ghi `install.log`, `pip-freeze.txt`, `pip_check.txt`, cấu hình `RUN_ENV` (`HF_TOKEN`, `USE_TF=0`).
- **Bước 6:** Subprocess preflight: xác minh `verify_hf_model_metadata(MODEL_NAME, verify_license=True)`, lưu revision SHA và license vào `environment.json` cùng thông tin GPU/VRAM/precision.
- **Bước 7:** Chạy smoke `dev-only` (`max_examples=20`) vào thư mục riêng `/content/task11_llm_smoke`.
- **Bước 8:** Chạy `dev-sweep` (`vsmec`, `neuesc`) rồi `eval-test` (`vsmec`, `neuesc`) qua subprocess với cơ chế `SKIP` từng tổ hợp + đồng bộ Drive phẳng ngay sau mỗi tổ hợp/bước hoàn tất (đồng thời giữ mô hình ở ngoài kernel chính để không tràn VRAM).
- **Bước 9:** Kiểm tra `SKIP` và nghiệm thu độc lập toàn bộ `dev_sweep_*.csv`, `selection_llm_*.json`, `predictions_test.csv` (`ZS_PRED_COLUMNS`, `in_strict=526`, tổng xác suất = 1), `metadata.json` (`pretraining_leakage_risk=True`) và `frozen_before.json`.
- **Bước 10:** Chạy `benchmark` chi phí LLM cho cả `zero_shot` và `few_shot` đã khóa, gọi `build_task11_tables_and_report` (tự nhận diện dữ liệu `Task08` và `Task11_XLMR` trên Drive nếu đã có để ghép bảng đối sánh `ViSoBERT` – `XLM-R` – `LLM`), đồng bộ và kiểm chứng SHA-256 trên Drive phẳng.

### Pha 6 — Bổ sung Kiểm thử Offline CPU & `.gitignore`

1. **Tạo mới `tests/test_phase2_task11_xlmr.py`** (chạy CPU offline, mô hình BERT nhỏ local + `FakeTrainer`, không mạng/GPU):
   - Kiểm tra khóa LR 2 giai đoạn (`methods=("dft",)` → `selection_xlmr_dft_lr.json` → `DFT 100%` nhận `--selection-path` → `methods=("dft", "seq")` → `selection_xlmr_lr.json`).
   - Kiểm tra `_locate_phase2_xlmr_source_checkpoint`: từ chối thư mục `sweep`/`smoke`, từ chối run bị cắt `max_examples`, từ chối run thiếu `completion.json` hoặc head sai số lớp; xác nhận provenance `XLMR` không gán `unverified`.
   - Kiểm tra `run_xlmr_zs_eval`: đúng view `zsA`/`zsB`, kiểm tra `check_underscore_leakage`, `SKIP` khi đủ artifact và chạy lại khi xóa file.
   - Kiểm tra `benchmark_encoder_inference` trên CPU và `build_task11_tables_and_report` (đặc biệt xác nhận `V3-VSMEC-strict` trong `task11_per_class_f1.csv` chỉ có `negative` & `positive`, mọi contrast trong `task11_exploratory_contrasts.csv` đều `exploratory=True`).
2. **Mở rộng [`tests/test_phase2_task11_llm.py`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/tests/test_phase2_task11_llm.py):**
   - Kiểm tra prompt vượt `max_input_length`: ví dụ few-shot được rút gọn, phần câu truy vấn đích không bị cắt mất; ném lỗi nếu riêng câu truy vấn vượt trần.
   - Kiểm tra `parse_llm_label`: từ chối câu phủ định (`"Không phải positive"`), câu chứa nhiều nhãn (`"Sadness hoặc Anger"`, `"negative / positive"`) với `valid=False`.
   - Kiểm tra `SKIP` ở `dev-sweep` và `eval-test`, từ chối `eval-test` nếu `model_name` hoặc `precision` khác với `selection_llm_*.json`.
   - Kiểm tra đếm tham số `num_parameters()` (mock mô hình 4-bit có `numel()` đóng gói một nửa) và `benchmark_llm_inference` cho cả `zero_shot` & `few_shot` với `quiet=True`.
3. **Cập nhật [`.gitignore`](file:///c:/Users/ADMIN/PycharmProjects/Xdomain-phase2-task11/.gitignore):**
   - Thêm ngoại lệ `!tests/test_phase2_*.py` (khi nhóm chuẩn bị commit) để các bài test Phase 2 được đồng bộ lên GitHub/Colab.

### Pha 7 — Kiểm chứng trước khi chạy Colab GPU

Sau khi hoàn tất Pha 1–6 (ở bước thực thi sau khi được phê duyệt), bắt buộc chạy đủ các kiểm tra cục bộ:
1. Chạy toàn bộ suite test trong môi trường `.venv` của dự án (với `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`):
   ```powershell
   .venv/Scripts/python.exe -X utf8 -m pytest -q
   ```
2. Kiểm tra biên dịch cú pháp (`py_compile`) cho toàn bộ module Python và toàn bộ block code Python trích xuất từ hai notebook `.ipynb`.
3. Kiểm tra CLI `--help` cho `src.phase2.cli`, `src.phase2.select_lr`, `src.phase2.llm`, `src.phase2.task11` (không khởi chạy huấn luyện hay tải mô hình).
4. Xác minh toàn vẹn mã băm của mọi tài nguyên đóng băng (`data/clean/`, `data/splits/`, `data/subsets_phase2/`, `configs/seq.yaml`, `configs/phase2.yaml`, `src/common/constants.py`, `src/seq/`, `src/zs/`, `src/stats/`) hoàn toàn không thay đổi.

---

## 5. Bảng kiểm các bất biến khoa học & Rủi ro cần canh gác

| Hạng mục | Quy tắc bắt buộc |
|---|---|
| **Dữ liệu & cấu hình đóng băng** | Không sửa `data/clean/`, `data/splits/`, `data/subsets_phase2/`, `HARD_MERGE_RULE`, `configs/seq.yaml`, `configs/phase2.yaml`, `src/seq/`, `src/zs/`, `src/stats/`. |
| **Cách ly `test`** | Không dùng `test` để chọn LR cho `XLM-R`, không dùng `test` để chỉnh prompt, chọn `prompt_style` hay chiến lược few-shot cho `LLM`. |
| **Gia đình `Holm m = 2`** | Chỉ dành cho 2 so sánh chính của `ViSoBERT` ở 10% (Task 08). Toàn bộ phép so sánh liên quan `XLM-R` và `LLM` phải gắn `primary=False, exploratory=True`. |
| **Ghép cặp `Seq − ZS`** | Trên chiều `SB`, chỉ ghép trên các góc nhìn 3 lớp (`V3-VSMEC`, `V3-VSMEC-strict`); tuyệt đối không ghép lệch `V7` với `V3`. |
| **Quy tắc `V3-VSMEC-strict`** | Điểm tổng hợp chấm trên nhãn `{0, 2}` (dự đoán nhầm vào nhãn trung tính `1` vẫn là lỗi); bảng `per_class_f1` **chỉ xuất 2 dòng `negative` và `positive`**. |
| **Rủi ro rò rỉ LLM** | Mọi metadata và báo cáo LLM bắt buộc ghi nhận `pretraining_leakage_risk=True` đối với `UIT-VSMEC` và `NEU-ESC`. |
| **Đóng băng `source_code_sha256`** | Phải commit xong toàn bộ thay đổi trong `src/phase2/*.py` trước khi chạy lượt GPU đầu tiên của Task 11 trên Colab để không làm lệch `fingerprint` giữa các lượt. |
