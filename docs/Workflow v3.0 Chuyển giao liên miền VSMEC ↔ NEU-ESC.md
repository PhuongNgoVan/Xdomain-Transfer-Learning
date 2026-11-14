# Workflow v3.0: Chuyển giao liên miền VSMEC ↔ NEU-ESC

> Bổ sung seq-v1, 15/11/2026: cấu hình thực thi ở [configs/seq.yaml](../configs/seq.yaml), quyết định ở [seq_notes.md](seq_notes.md). Patience 5, warmup 6% theo config baseline trong Git; chưa xác nhận log lượt thật. Sweep chỉ train/dev, test chỉ sau chọn best. Dùng trainer sạch, không phụ thuộc train_neu3.py cũ.


*Bản rút gọn của kế hoạch v3.0 (10/11/2026). Chỉ cần làm theo file này; chi tiết tra lại kế hoạch đầy đủ khi cần.*

## 0. Cố định (không bàn lại)

- **Câu hỏi:** chuyển giao hai chiều VSMEC → NEU và NEU → VSMEC có hiệu quả không, theo chiều nào, vì sao.
- **Backbone:** PhoBERT (vinai/phobert-base-v2), ViSoBERT (uitnlp/visobert).
- **Nhãn:** so sánh liên miền luôn trong không gian **3 nhãn**. Enjoyment → Positive; Sadness, Fear, Anger, Disgust → Negative; Surprise, Other → Neutral. NEU-ESC gộp toxic vào negative.
- **Cấu hình:** cross-entropy chuẩn, LR dò {2e-5, 3e-5, 5e-5}, batch 32, tối đa 10 epoch, early stop theo macro-F1 dev (patience 5), 5 seed.
- **Độ đo chính:** macro-F1 (phụ: weighted-F1, accuracy, F1 từng lớp).
- **Chọn mô hình:** chỉ dùng dev của miền đang huấn luyện. Test chỉ để báo cáo.

## 1. Chuẩn bị dữ liệu (tuần 1–2)

1. NFC, thu gọn ký tự lặp, giữ emoji; PhoBERT tách từ bằng VnCoreNLP.
2. Khử trùng lặp trong từng split; câu trùng chéo split loại khỏi train.
3. Câu trùng giữa VSMEC và NEU-ESC: loại khỏi **test** cả hai miền, báo cáo số lượng.
4. max\_length 256, dynamic padding; chia 80/10/10 phân tầng nếu không có split chính thức; lưu file chỉ mục và hash.
5. Gán nhãn thủ công \~100 câu Other/Surprise (2 người, Cohen's kappa); đo VSMEC-strict.
6. **Đóng băng dữ liệu và quy tắc ánh xạ cuối tuần 2.**

## 2. Chạy thử toàn chuỗi (tuần 3)

- 1 seed, 1 backbone chạy đủ Bước 1 → 2 → 3; đo thời gian thực; chốt 4 so sánh chính. **Go/no-go cấu hình.**

## 3. Bước 1: Baseline (tuần 4–5)

- Train độc lập trên VSMEC (V7) và NEU-3: 2 backbone × 2 miền × 5 seed = 20 lượt, lưu 20 checkpoint.
- Tính thêm V3-VSMEC bằng ánh xạ (không train lại); mốc sàn majority/ngẫu nhiên.
- **Cổng qua bước:** F1 VSMEC khớp số đã công bố, có std giữa các seed. Lệch nhiều thì rà lại pipeline.

## 4. Bước 2: Zero-shot (tuần 6)

- Chiều A: checkpoint VSMEC chấm trên test NEU (cộng xác suất 7 → 3). Chiều B: checkpoint NEU chấm trên test VSMEC (đầy đủ và strict).
- Không fine-tune, không chỉnh ngưỡng bằng dữ liệu đích. Chỉ suy luận: 20 lượt.
- Báo cáo transfer gap, retention, bất đối xứng, JS divergence nhãn dự đoán, ma trận 7 × 3.
- Độ lệch nội tại: bề mặt (độ dài, OOV, emoji), JS tiên nghiệm nhãn, PAD + t-SNE/UMAP.
- Chẩn đoán: oracle prior correction, độ nhạy ánh xạ, theo nhóm câu → bảng quy nguyên nhân (nhãn/ánh xạ, tiên nghiệm, lệch ngôn ngữ).

## 5. Bước 3: Sequential (tuần 7–8)

- S-A: VSMEC → NEU. S-B: NEU → VSMEC.
- Giai đoạn 1: dùng lại checkpoint Bước 1 cùng seed. Giai đoạn 2: giữ encoder, đầu mới ngẫu nhiên, fine-tune toàn bộ trên đích; LR dò {1e-5, 2e-5, 3e-5} trên dev đích.
- 2 backbone × 2 chiều × 5 seed = 20 lượt; so sánh có cặp với baseline cùng seed.
- Đo: Gain, forgetting (linear probe), CKA hoặc khoảng cách embedding.
- **Thống kê:** 4 so sánh chính (2 chiều × 2 backbone), paired bootstrap, hiệu chỉnh Holm, mức 0,05. Mọi so sánh khác là thăm dò.

## 6. Đối chứng và mở rộng (tuần 9–11)

| Ưu tiên | Việc |
| --- | --- |
| Cao | C0 self-sequential; V1 shared-3 |
| Trung bình | V2 joint |
| Mở rộng | V4 đích 10%, V3, V5, V6 (PhoBERT-large) |

- Go/no-go cho phần mở rộng ở tuần 9. Thứ tự cắt khi thiếu giờ: V6 → V3 → V5 → V4 → giảm seed.

## 7. Báo cáo (viết dần từ tuần 4)

- Phân tích lỗi: F1 từng lớp, ma trận nhầm lẫn, nhóm câu, ví dụ đúng/sai.
- Kết quả dương, bằng 0 hay âm đều hợp lệ nếu được phân tích nguyên nhân.
- Nhớ: test VSMEC nhỏ (\~700 câu), khoảng tin cậy rộng, không kết luận từ chênh lệch vài điểm.
- Tuần 12–13 hoàn thiện, tuần 14 đệm và chuẩn bị bảo vệ.

## 8. Quy tắc vàng

1. Không dùng dữ liệu đích (kể cả dev) ở Bước 2.
2. Mỗi run ghi một dòng CSV: run\_id, backbone, stage1, stage2, eval\_domain, label\_view, mode (base/zs/seq), fraction, seed, macro\_f1, weighted\_f1, accuracy, f1\_per\_class, ckpt\_path.
3. Tên model: `{Backbone}_{GĐ}` (vd `PhoBERT_VSMEC>NEU`). Lưu YAML cấu hình, seed, commit, hash dữ liệu.
4. Một repo, một bộ tiền xử lý, một script đánh giá; lưu checkpoint thường xuyên.
5. Số liệu NEU-ESC trong kế hoạch phải tính lại từ split thực tế trước khi đưa vào báo cáo.

## 9. Ngân sách

- Lõi \~65 lượt train (12 dò LR + 20 Bước 1 + 12 dò LR GĐ2 + 20 Bước 3; Bước 2 chỉ suy luận). Làm hết mở rộng \~125 lượt.
