# vi-emo2sent

Chuyển giao 7 cảm xúc (UIT-VSMEC) sang 3 nhãn quan điểm (NEU-ESC) và đánh giá ngoài miền trên UIT-VSFC.

## Cài đặt môi trường

Yêu cầu Python 3.10+. Khuyến nghị sử dụng môi trường ảo (`.venv`):

```bash
# Tạo và kích hoạt môi trường ảo (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Hoặc trên Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện đã khóa phiên bản
pip install -r requirements.txt
```

## Dò `lr × precision` cho Stage 1 (UIT-VSMEC 7 lớp)

Nhánh `visobert-vsmec-7c` hỗ trợ dò tổ hợp `learning_rate ∈ {2e-5, 3e-5, 5e-5}` × `precision ∈ {fp16, bf16}` × `seed ∈ {0, 1, 2, 3, 4}` (30 lượt) cho ViSoBERT Stage 1 trên cột đầu vào `text` kèm biến đổi trong bộ nhớ `emoji_mode: keep`:

- **Các khóa cấu hình mới trong YAML Stage 1**:
  - `precision`: hỗ trợ `"fp16"`, `"bf16"`, `"fp32"` (giá trị khác báo `ValueError`). Đánh giá dev và xuất dự đoán (`predictions7/`) luôn chạy ở FP32.
  - `strict_precision` (mặc định `false`): khi `true`, nếu môi trường không đáp ứng được `precision` yêu cầu (ví dụ không có CUDA hoặc GPU không hỗ trợ `bf16` theo `torch.cuda.is_bf16_supported()`), script lập tức bắn `RuntimeError` thay vì tự hạ xuống `fp32`. Khi `false`, cho phép fallback sang `fp32` và ghi nhận độ chính xác thực tế vào `precision_effective`.
  - `emoji_mode` (mặc định `"raw"` khi vắng mặt để tương thích ngược):
    - `"raw"` (hoặc bỏ trống): đọc nguyên cột gốc (`text` cho ViSoBERT, `text_seg` cho PhoBERT).
    - `"keep"` (chỉ cho ViSoBERT): áp dụng `preprocess_text(str(t), emoji_mode="keep")` hoàn toàn trong bộ nhớ (in-memory) trên cả 6 tập (`vsmec_train`, `vsmec_dev`, `vsmec_test`, `neuesc_dev`, `neuesc_test`, `vsfc_test`) mà không chỉnh sửa dữ liệu trên đĩa. Giá trị khác hoặc dùng `"keep"` cho PhoBERT sẽ báo `ValueError`.
  - `sweep` (mặc định `false`): khi `true`, không ghi đè file chính thức `results/stage1_ckpt_hash.json` mà ghi mã băm SHA-256 của checkpoint vào `results/ckpt_hash_sweep.json` (khóa theo `run_id`).
  - `save_checkpoint` (mặc định `true`): khi `false`, bỏ qua bước lưu trọng số vật lý (`encoder/` + `head.pt`) và băm checkpoint trên đĩa, nhưng vẫn nạp `best_state_dict` trong RAM để xuất đầy đủ 5 file `predictions7/`, dòng kết quả trong `stage1.csv` và file sidecar.
  - `variant`: định danh biến thể trong `run_id` theo mẫu `{alias}_Stage1-{variant}_s{seed}_lr{lr}` (ví dụ `ViSoBERT_Stage1-emokeep-bf16_s0_lr3e-5`). Giá trị `variant` không được rỗng và **không được chứa dấu gạch dưới (`_`)**.
- **File chẩn đoán sidecar (`<results_dir>/runs/<run_id>.json`)**:
  - Được ghi tự động sau mỗi lượt hoàn tất (sau khi đã ghi 5 file `predictions7/` và dòng `stage1.csv`).
  - Gồm các trường: `run_id`, `alias`, `backbone`, `lr`, `seed`, `variant`, `precision`, `precision_effective`, `emoji_mode`, `transformed_text_sha256` (SHA-256 của văn bản sau biến đổi từng tập), `git_commit`, `data_hash`, `torch_version`, `transformers_version`, `gpu`, `best_step`, `best_dev_macro_f1`, `elapsed_sec`, `truncation_report` (kèm tỉ lệ cắt cụt mới và `text_sha256`), `eval_history` (danh sách `{step, epoch, dev_macro_f1, dev_loss, dev_accuracy}` của mọi lần đánh giá dev), `n_skipped_optimizer_steps`, `n_nonfinite_loss`, và `report_only` (`{dev_macro_f1, test_macro_f1}`).
- **Quy tắc kỷ luật chọn mô hình**:
  - Việc chọn tổ hợp `(lr, precision)` và chọn checkpoint tốt nhất **chỉ được dựa trên tập `vsmec_dev`** (cột `macro_f1` trong `stage1.csv` hoặc `best_dev_macro_f1` / `eval_history`).
  - Trường `report_only.test_macro_f1` trong file sidecar chỉ phục vụ báo cáo phân tích độ lệch `dev - test` sau khi đã đóng băng lựa chọn vào `selection.json`; **tuyệt đối không dùng tập test để chọn siêu tham số hay chọn checkpoint**.




