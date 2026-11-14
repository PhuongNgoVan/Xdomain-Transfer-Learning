# Phương án sequential đã chốt — 15/11/2026

Nguồn cấu hình duy nhất là `configs/seq.yaml` (seq-v1); hợp đồng hiện hành
là `tasks/AGENT_step3_seq.md`. Dùng trainer sạch trong `src/seq/`, tái sử
dụng model_ckpt, locator và label views đã kiểm thử. Không còn điều kiện
phải có hoặc chọn/copy `train_neu3.py` cũ đang lỗi self-import.

## Cấu hình và provenance

Batch16×2, epochs10, patience5, warmup0.06, weight_decay0.01, clip1,
dropout0.1, max_length256, dynamic padding, nhóm theo độ dài, AdamW không
decay bias/LayerNorm, linear scheduler; đánh giá hai lần/epoch theo bước
optimizer, chọn best bằng macro-F1 dev. Head mới có seed bằng seed lượt.
Đây là cấu hình baseline trong Git, chưa xác minh với log các lượt thật.
Giữ `baseline_provenance.actual_run_verified=false` tới khi có bằng chứng;
nếu baseline thật khác thì chạy lại cùng giao thức hoặc báo nhiễu cấu hình.
Không sửa điểm số lịch sử để phù hợp config. Giao thức này bổ sung v3.0.

BF16 CUDA strict cho nghiên cứu; không fallback FP32 âm thầm. Unit test
dùng trainer giả; test AMP CPU đặt strict_precision=false tường minh.
CPU test không chứng nhận BF16 CUDA hay tái lập baseline nghiên cứu.

## Chọn LR và test

Sweep SA/SB × hai backbone × LR1e-5/2e-5/3e-5, seed0: chỉ train/dev,
không mở/tokenize/chấm test; một dòng dev view chính, không checkpoint cuối.
NEU chung một CSV: chỉ lọc train/dev để xử lý, hash byte toàn file để định
danh. select_lr yêu cầu đủ lưới, cùng data/code, hòa chọn LR nhỏ hơn.
Main/C0 bắt buộc dùng selection_seq.json. C0 dùng lại LR SA/SB tương ứng,
seed0..2; seq-v1 không giữ tùy chọn dò C0 chưa có giao thức.
Test chỉ xử lý sau khi fit khôi phục best dev. PhoBERT dùng text_seg;
ViSoBERT text/emoji keep ở mọi variant, không suy từ tên variant.

## Checkpoint và phục hồi

Mỗi run có checkpoint riêng theo run_id, fingerprint config/data/source
weights/commit và sidecar runs_seq. SKIP chỉ khi đủ dòng và hash tất cả
artifact khớp. Resume lưu best weights, cursor batch, RNG Python/NumPy/
Torch/CUDA, optimizer/scheduler/history. Giữ trạng thái resume để tái tạo
artifact nếu bị ngắt khi xuất kết quả. Không sửa dữ liệu/checkpoint nguồn.

## Thống kê và giới hạn

Sequential ghép baseline cùng seed theo thiết kế, không nói cùng checkpoint
nguồn (SA/SB có nguồn khác baseline đích). Bootstrap ghép câu và seed ID
cùng thứ tự; C0 dùng giao seed. Mặc định zero-shot không đổi.
Holm giữ đúng bốn primary và năm seed mỗi so sánh; thiếu thì p_holm và
significant trống, không giảm m. View phụ, C0, weighted-F1, accuracy và
F1 từng lớp là thăm dò. seq_vs_zs chỉ ghép cùng view3 lớp, không ghép V7/V3.

Xem `seq_metric_definitions.md`, `seq_primary_comparisons.md` để biết
công thức p_boot và giới hạn diễn giải. Chưa truy cập Drive, chạy dữ liệu
nghiên cứu hay xác nhận cấu hình/điểm baseline thật. Mã chưa commit/push.

## Trình tự dùng sau này

1. Chuẩn bị checkpoint Bước 1 và prediction tham chiếu Bước 2; đối chiếu
   config/log baseline thật. Commit cấu hình và bốn primary trước lượt thật.
2. Chạy 12 sweep bằng `python -m src.seq.train` với
   `--variant seqsweep-bf16`, seed0, hai backbone, SA/SB và ba LR trong config.
3. Chọn LR: `python -m src.seq.select_lr --results-dir <results>`.
   Không tự điền LR chính dựa trên điểm test.
4. Chạy main/C0 với LR từ selection_seq.json, mỗi run có checkpoint riêng.
   Dùng thư mục kết quả riêng cho smoke test (`--max-examples`).
5. Phân tích: `python -m src.seq.analyze --results-dir <results>
   --zs-results-dir <step2-results> --data-dir data/clean --out-dir <tables>`.
   Khi thiếu dữ liệu, `--allow-missing` cho bảng tạm nhưng chưa có kết luận Holm.

## Xác minh triển khai

Ngày 15/11/2026: **75 test PASS** toàn repo, gồm zero-shot cũ và sequential
mới; py_compile sạch và ba CLI --help chạy được. Kiểm thử gồm mô hình BERT
nhỏ local, BF16 CPU tường minh, gián đoạn giữa epoch rồi resume, sweep
không chấm test, artifact thiếu không SKIP, lưới LR và Holm thiếu gia đình,
toàn bộ bảng trên dữ liệu giả cho hai backbone × năm seed × hai chiều,
và tái lập nguyên byte khi phân tích lại. Không chứng nhận CUDA/Drive hoặc
các lượt baseline thật từ những kiểm thử này.
