# Hợp đồng Bước 3 — seq-v1

Làm trên seq-core. Đọc docs/seq_notes.md, seq_metric_definitions.md và
seq_primary_comparisons.md. Config duy nhất configs/seq.yaml. Thay thế bản(1).

## 1. Kiến trúc và phạm vi

Trainer sạch src/seq; dùng Stage1Classifier, locator/views/mapping hiện có.
Không yêu cầu/copy train_neu3/stage1 cũ. Giữ nguyên dữ liệu, hằng số nhãn,
src/zs và test cũ. Mở rộng bootstrap tương thích ngược; thêm SEQ_COLUMNS.
Không truy cập Drive hoặc chạy lượt nghiên cứu thật trong lúc triển khai.
SA: VSMEC7→NEU3; SB: NEU3→VSMEC7; C0N/C0V cùng miền. Head mới seed=seed,
fine-tune toàn bộ. PhoBERT text_seg; ViSoBERT text/emoji keep mọi variant.

## 2. Cấu hình

Batch16×2, epochs10, patience5, warmup0.06, weight_decay0.01, clip1,
dropout0.1, max_length256, dynamic padding/nhóm theo độ dài; AdamW không
decay bias/LayerNorm, linear scheduler. BF16 CUDA strict, không fallback.
Cấu hình baseline trong Git không phải xác minh lượt thật; xuất provenance.

## 3. Sweep và LR

seqsweep-bf16: seed0, LR{1e-5,2e-5,3e-5}, SA/SB × hai backbone, đủ12lượt.
Chỉ train/dev, một dòng dev view chính + prediction dev; không test hoặc
checkpoint cuối. NEU chung CSV: lọc trước xử lý, hash byte chỉ định danh.
select_lr yêu cầu đủ lưới, seed0, view chính, cùng hash/code, chọn max dev
rồi min LR; xuất selection_seq.json và seq_lr_selection.csv.
Main seed0..4; C0 seed0..2, reuse LR tương ứng SA/SB. Không hỗ trợ dò C0
trong seq-v1. Main/C0 phải khớp selection và protocol hash.

## 4. Huấn luyện và artifact

run_id={alias}_{TAG}-{variant}_s{seed}_lr{lr}; TAG=SEQV2N/SEQN2V/C0N/C0V;
variant không underscore. run_stage2(trainer=None) cho phép test giả.
CLI src.seq.train: --config --backbone --seed --direction --lr --variant
--vsmec-ckpt-root --neu-ckpt-root --data-dir --results-dir --ckpt-out-dir
[--max-examples N] [--no-save-ckpt].
Test chỉ xử lý sau fit khôi phục best dev. Prediction dùng ZS_PRED_COLUMNS;
score từ file đã ghi. Main/C0: NEU dev/test V3; VSMEC dev V7, test V7 cộng
V3-full/strict ×sum/argmax. SEQ_COLUMNS giữ metadata/metrics theo schema.
Checkpoint {ckpt_out_dir}/{run_id}, sidecar runs_seq và fingerprint config,
data/source/code. Resume lưu cursor/RNG/optimizer/scheduler/best/history.
SKIP yêu cầu đủ dòng và hash mọi artifact khớp; CSV đơn độc không đủ.

## 5. Phân tích

CLI src.seq.analyze: --results-dir --zs-results-dir --data-dir --out-dir
[--config] [--boot] [--seed] [--allow-missing]. Baseline REFN2N/REFV2V;
tính lại từ prediction cùng ID/nhãn/seed. share_seeds=True phải cùng thứ
tự ID seed; default False giữ kết quả cũ. C0 dùng giao seed.
Xuất seq_main.csv, seq_per_class.csv (strict chỉ neg,pos), seq_c0.csv,
seq_lr_selection.csv, seq_confusion_{backbone}_{direction}_{kind}_{counts|rownorm}.csv,
seq_vs_zs.csv (chỉ cùng view3), seq_meta.json (gồm provenance/commit primary).
kind=3x3 cho NEU;7x7 và7x3 cho VSMEC. Chỉ bốn macro-F1 primary được Holm
m=4 khi đủ năm seed mỗi so sánh; thiếu để p_holm/significant trống.
Các metric/view/C0 khác thăm dò. Không thay thống kê p_boot đã định nghĩa.

## 6. Nghiệm thu

CPU/offline: đầu mới, encoder nguyên vẹn, seed; test không tham gia chọn
LR/best; BF16 CPU explicit; bốn hướng và sweep dev-only; grid/tie/edge;
hash/resume/SKIP; paired mismatch/default bootstrap không đổi; Holm thiếu
gia đình; C0; analyzer đủ10cặp hai hướng và tái lập nguyên byte.
Pytest cũ xanh, py_compile sạch. Ghi rõ chưa xác minh CUDA/Drive/baseline
thật. Không tự commit/push hoặc train nghiên cứu thật.
