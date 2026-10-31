
# Quy ước bắt buộc cho mọi model

Căn cứ: Kế hoạch NCKH v2.2 (27/10/2026) và bảng phân công nhóm 3 người v2.

---

## 1. Dữ liệu và chia tập

- **NEU-ESC**: đích huấn luyện và đánh giá chính. Nhãn gốc toxic được gộp vào negative (6.045/32.966 mẫu).
- **UIT-VSMEC**: chỉ dùng huấn luyện 7 lớp (Stage 1, nhóm B2).
- **UIT-VSFC**: chỉ dùng tập test để đánh giá ngoài miền (OOD). Không huấn luyện, không dùng để chọn model hay siêu tham số. Chỉ dùng bản test đã làm sạch (không trùng lặp, không mâu thuẫn nhãn, không trùng với NEU-ESC và VSMEC).
- **Mốc dữ liệu NEU-ESC**: 100%, 10%, 5%, 1%, lấy phân tầng theo nhãn.
- **Tập dev thu nhỏ cùng tỉ lệ** (phân tầng). Tập test luôn giữ nguyên đầy đủ.
- **Mốc 5% là mốc so sánh trọng tâm.** Thứ tự ưu tiên chạy: 5% → 100% → 1% → 10%.
- Các file split được đóng băng ở tuần 3. Mọi model dùng chung đúng các file này, không tự chia lại. Lưu hash từng file dữ liệu và phiên bản mã tiền xử lý.

## 2. Nhãn

Thứ tự nhãn thống nhất cho toàn bộ dự án:

| Chỉ số | Nhãn |
| --- | --- |
| 0 | negative |
| 1 | neutral |
| 2 | positive |

- Nhãn gốc NEU-ESC là 0 neutral, 1 positive, 2 negative, 3 toxic. Ánh xạ **đúng một lần** ở bước tiền xử lý: gốc 2 và 3 → 0, gốc 0 → 1, gốc 1 → 2. Sau bước này không ai dùng lại nhãn gốc.
- Mọi cột xác suất (`p_neg, p_neu, p_pos`) và metric từng lớp theo đúng thứ tự trên.
- VSMEC giữ 7 nhãn: Enjoyment, Sadness, Fear, Anger, Disgust, Surprise, Other.

**Luật gộp nhãn cứng (nhóm B2, các model Hard_\*):**

| VSMEC | Nhãn 3 lớp |
| --- | --- |
| Enjoyment | positive |
| Sadness, Fear, Anger, Disgust | negative |
| Other, Surprise | neutral |

Luật này có thể đổi nếu kết quả kiểm chứng cực tính Other/Surprise (kế hoạch mục 6.2) không nghiêng về neutral. Khi đó cập nhật cho tất cả model Hard_\* cùng lúc.

## 3. Tiền xử lý thống nhất

- Chuẩn hóa Unicode NFC, hạ chữ thường, thu gọn ký tự lặp, giữ emoji hoặc ánh xạ sang token.
- Tách từ bằng VnCoreNLP cho PhoBERT. Dùng tokenizer gốc cho ViSoBERT và XLM-R.
- `max_length = 256` token (tính theo token, không theo từ), kèm dynamic padding (`DataCollatorWithPadding`) và `group_by_length=True`.
- Ghi lại tỉ lệ mẫu bị cắt cụt theo từng tokenizer.
- Câu ≤ 2 từ giữ nguyên trong cấu hình chính.

## 4. Seed

- Bộ seed cố định dùng chung cho mọi model: `{42, 43, 44, 45, 46}`.
- **Mốc 100% và 5%**: 5 seed (42–46). **Mốc 10% và 1%**: 3 seed (42–44).
- **Một seed điều khiển cả hai việc**: lấy tập con 1/5/10% và khởi tạo huấn luyện. Cùng seed thì mọi model nhận đúng cùng tập con, nên so sánh B5 với B4-LA ghép cặp được, phục vụ kiểm định bootstrap hai tầng.
- Hạn chế cần ghi vào báo cáo: phương sai do chọn tập con và do khởi tạo bị gộp làm một.
- Model chỉ có một lượt (Hard_\*, Eval Test) ghi `seed = NA`.

## 5. Huấn luyện

Mọi PLM dùng cùng siêu tham số cơ sở, cùng seed và cùng ngân sách dò siêu tham số.

| Tham số | Giá trị |
| --- | --- |
| Batch size hiệu dụng | 32 (16 × 2 gradient accumulation) |
| Learning rate | Dò {2e-5, 3e-5, 5e-5}, xem quy trình dò bên dưới |
| Epoch | 5–10 |
| Warmup | 6–10% số bước |
| Weight decay | 0,01 |
| Độ chính xác | bf16 trên A100, fp16 trên GPU Kaggle |

**Quy trình dò learning rate:**
- Chỉ dò ở mốc 100%, **seed 42**, chọn theo **macro-F1 trên dev đầy đủ** (3.305 mẫu). Không nhìn test.
- Dò trên B3 của từng backbone (PhoBERT_Tgt, ViSoBERT_Tgt), rồi dùng chung cho B4, B4-LA, B4-init, B5 cùng backbone.
- Ngân sách dò: 3 lượt × 2 backbone = 6 lượt (cần cộng vào ước lượng GPU, kế hoạch gốc chưa tính).
- Giá trị learning rate cuối cùng ghi vào mục này sau khi dò xong: PhoBERT = `___`, ViSoBERT = `___`.

**Quy tắc dừng sớm theo mốc:**

| Mốc | Quy tắc |
| --- | --- |
| 100% | Dừng sớm theo macro-F1 trên dev đầy đủ |
| 10% và 5% | Dừng sớm theo val_loss (có hiệu chỉnh LA), kèm số bước tối thiểu |
| 1% | Không dừng sớm. Dùng số bước cố định lấy từ cấu hình mốc 5% |

- Chọn model và siêu tham số **chỉ dựa trên dev** của NEU-ESC.
- Tập test chỉ dùng một lần để báo cáo, và chỉ chạy sau khi đã chốt checkpoint theo dev.

## 6. Logit Adjustment (B4-LA, B4-init, B5)

- Dùng chung một module LA duy nhất do Phương cung cấp, không tự cài lại.
- **τ = 1,0 cố định** ở mọi mốc ít dữ liệu. Chỉ quét τ ∈ {0,5; 1,0; 1,5; 2,0} ở mốc 100% như ablation.
- π_y tính trên đúng tập huấn luyện của mốc đang chạy.
- Lưu dự đoán ở **hai chế độ suy luận**, không cần huấn luyện lại:
  - `bal`: bỏ độ lệch lúc suy luận, cho bộ phân loại cân bằng.
  - `prior`: giữ một phần hiệu chỉnh theo tiên nghiệm của tập test.

## 7. Đánh giá

- **Độ đo chính**: macro-F1.
- **Độ đo phụ**: accuracy, weighted-F1, F1 từng lớp, balanced accuracy, confusion matrix. Thêm ECE khi so LA với các phương án khác.
- Mỗi checkpoint được đánh giá trên ba tập: dev NEU-ESC, test NEU-ESC và test UIT-VSFC (OOD).
- Với UIT-VSFC báo cáo F1 từng lớp và confusion matrix, không chỉ macro-F1, vì neutral lệch rất mạnh (~4% so với ~69%).
- **Nhóm B2 (Hard_\*)** chỉ chạy Eval Test: huấn luyện 7 lớp trên VSMEC, gộp theo luật ở mục 2, rồi đánh giá trên test NEU-ESC và UIT-VSFC. Nên báo cáo thêm macro-F1 7 lớp trên VSMEC.
- **B4-init**: đo macro-F1 của đầu khởi tạo trên dev NEU-ESC **trước khi** fine-tune và ghi lại.

## 8. Quy ước đặt tên model

Dạng `[Backbone hoặc đặc trưng]_[Phương pháp]_[Kỹ thuật bổ sung]`. Dùng thống nhất cho bảng kết quả, tên file cấu hình, checkpoint và tên run (W&B/MLflow, Kaggle).

- Backbone: `PhoBERT`, `ViSoBERT`, `XLMR`. Đặc trưng cổ điển: `TF`.
- Phương pháp: `Tgt`, `Hard`, `HR`, `GESF`.
- Kỹ thuật bổ sung: `LA`, `Init`, `Emo`.
- Ví dụ: `TF_SVM`, `ViSoBERT_Tgt`, `PhoBERT_HR_LA`, `ViSoBERT_GESF_LA`.
- **Mốc dữ liệu và seed không đưa vào tên model**, ghi ở cột riêng trong file kết quả.

## 9. File dự đoán

Mỗi lượt chạy lưu một file cho mỗi tập đánh giá.

- **Thư mục**: `outputs/preds/{model}/`
- **Tên file**: `{model}__{dataset}__f{fraction}__s{seed}.csv`
  - Ví dụ: `ViSoBERT_HR_LA__neu_esc_test__f5__s42.csv`
  - `dataset` ∈ `neu_esc_dev`, `neu_esc_test`, `uit_vsfc_test`
  - `fraction` ∈ `1`, `5`, `10`, `100`
- **Cột**: `id, y_true, y_pred, p_neg, p_neu, p_pos`
  - `id` lấy từ file split đã đóng băng, không tự đánh số lại.
  - `y_true`, `y_pred` theo thứ tự nhãn ở mục 2.
- **Model có Logit Adjustment**: lưu hai file, thêm hậu tố `__bal` hoặc `__prior` trước đuôi `.csv`.
- **Model chỉ có một lượt (Hard_\*)**: dùng `f100` và `sNA`.
- **Cột mở rộng (tùy mô hình)**: sáu cột trên là cột lõi, bắt buộc và đứng đầu theo đúng thứ tự. Mỗi mô hình được thêm cột riêng **sau** cột lõi, tên cột viết thường, không dấu, không trùng tên cột lõi. Script đánh giá chỉ đọc cột lõi và bỏ qua cột thêm. Ví dụ:
  - GESF: `alpha` (giá trị cổng), `p_emo_enj, p_emo_sad, p_emo_fear, p_emo_anger, p_emo_disg, p_emo_surp, p_emo_other`
  - Hard_\*: `y_pred_7` (nhãn 7 lớp trước khi gộp)
  - Mọi model: `n_tokens`, `truncated` (câu có bị cắt do `max_length` không)

## 10. File kết quả chuẩn

Một CSV chung cho cả nhóm, ở **dạng dài** (mỗi dòng một metric). Script đánh giá chung chạy trên các file dự đoán rồi ghi vào CSV này. Không ai ghi tay.

| Cột | Giá trị |
| --- | --- |
| `model` | Tên model theo mục 8 |
| `dataset` | `neu_esc_dev`, `neu_esc_test`, `uit_vsfc_test` |
| `fraction` | `1`, `5`, `10`, `100` |
| `seed` | `42`–`46`, hoặc `NA` |
| `infer_mode` | `bal`, `prior`, `na` (model không dùng LA) |
| `metric` | `macro_f1`, `accuracy`, `weighted_f1`, `balanced_acc`, `ece`, `f1_neg`, `f1_neu`, `f1_pos` |
| `value` | Số thực |

- **Cột `metric` mở**: mỗi mô hình ghi các metric phù hợp với mình, không bắt buộc đủ cả danh sách trên. Ngoài tám metric chuẩn, có thể thêm metric riêng, ví dụ:
  - B4-init: `macro_f1_init` (đầu khởi tạo trước khi fine-tune)
  - Hard_\*: `macro_f1_vsmec_7` (macro-F1 7 lớp trên VSMEC)
  - GESF: `p_emo_entropy`, `alpha_mean`
  - Mọi model: `truncation_rate`
  
  Quy tắc: tên metric viết thường, không dấu, dùng gạch dưới; cùng một ý nghĩa thì cùng một tên giữa các mô hình; metric mới thêm vào danh sách tên dùng chung trong repository để không ai đặt trùng ý khác tên.
- Dạng dài được chọn vì các nhóm model có tập metric khác nhau, thêm metric sau không phải sửa cấu trúc, và gộp, lọc, vẽ bảng bằng pandas dễ hơn.
- Confusion matrix không nằm trong CSV này. Script tính lại từ file dự đoán khi cần.

## 11. Tái lập và kỷ luật chạy

- Mỗi lượt lưu: file YAML cấu hình, seed, phiên bản thư viện, commit mã nguồn, hash dữ liệu, checkpoint tốt nhất (chọn theo dev).
- Model dùng Stage 1 (B2, B4, B4-LA, B4-init, B5): ghi rõ hash checkpoint Stage 1 đã dùng.
- Log huấn luyện: train loss, val_loss, macro-F1 dev theo bước, epoch hoặc bước tốt nhất, tỉ lệ mẫu bị cắt, thời gian chạy và loại GPU.
- Lưu checkpoint thường xuyên vì phiên Kaggle và Colab có thể bị ngắt. Ưu tiên chạy nền (batch) thay vì giữ phiên tương tác.
- Dùng chung một repository, một bộ mã tiền xử lý, một script đánh giá. Không sửa script đánh giá cục bộ.
- Lượt chạy lỗi hoặc chạy lại: ghi chú trong Run Tracker. Mỗi bộ (model, fraction, seed) chỉ có một kết quả cuối.
- Một lượt coi là **hoàn thành** khi đủ bốn điều kiện:
  1. Có file dự đoán cho cả ba tập đánh giá.
  2. Đã có dòng tương ứng trong CSV kết quả.
  3. Có checkpoint và log.
  4. Đã cập nhật Run Tracker.

## 12. Còn mở

| Mục | Trạng thái |
| --- | --- |
| Learning rate PhoBERT và ViSoBERT | Chờ kết quả dò ở mốc 100% (mục 5) |
| Hạn chót nội bộ cho checkpoint Stage 1 và module LA của Phương | Cần nhóm thống nhất, vì đây là điểm nghẽn của B2, B4, B4-LA |
| Luật gộp nhãn Other/Surprise | Có thể đổi sau khi kiểm chứng cực tính (mục 2) |
