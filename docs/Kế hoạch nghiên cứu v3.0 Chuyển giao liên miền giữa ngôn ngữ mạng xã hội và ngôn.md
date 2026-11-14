# Kế hoạch nghiên cứu v3.0: Chuyển giao liên miền giữa ngôn ngữ mạng xã hội và ngôn ngữ học đường cho phân loại cảm xúc tiếng Việt

**Dữ liệu:** UIT-VSMEC (mạng xã hội, 7 cảm xúc), NEU-ESC (học đường, 3 nhãn quan điểm) | **Backbone:** PhoBERT, ViSoBERT | **Phiên bản:** 3.0, 10/11/2026 (thay thế các bản 2.x)

## 1. Hướng mới và khác biệt so với kế hoạch cũ

> Bổ sung seq-v1, 15/11/2026: [configs/seq.yaml](../configs/seq.yaml) là cấu hình thực thi duy nhất; [seq_notes.md](seq_notes.md) ghi quyết định. Patience 5, warmup 6% lấy từ config baseline trong Git, chưa chứng nhận log các lượt thật. Trainer sạch trong src/seq; sweep chỉ train/dev. Những điểm baseline lịch sử chỉ được coi tương đương giao thức sau đối chiếu config/log thật.

Đề tài chuyển từ việc đề xuất kiến trúc mới (GESF + Logit Adjustment) sang **nghiên cứu thực nghiệm có hệ thống về chuyển giao hai chiều giữa hai miền**, theo ba bước: Baseline → Zero-shot cross-domain → Sequential transfer. Trọng tâm là các mô hình ngôn ngữ tiền huấn luyện (PLM) PhoBERT và ViSoBERT. Các mô hình đơn giản (TF-IDF + LR/SVM, BiLSTM) chỉ là phần học tập của thành viên mới (mục 11).

| Hạng mục | Kế hoạch cũ (v2.2) | Kế hoạch mới (v3.0) |
| --- | --- | --- |
| Câu hỏi chính | GESF + LA có vượt Head-Replacement + LA không? | Chuyển giao giữa hai miền có hiệu quả không, theo chiều nào, vì sao? |
| Chiều chuyển giao | Một chiều VSMEC → NEU-ESC; chiều ngược để hướng phát triển | **Hai chiều** VSMEC → NEU và NEU → VSMEC là trọng tâm |
| UIT-VSFC | Đánh giá ngoài miền | Bỏ khỏi lõi; chỉ là miền thứ ba nếu còn thời gian |
| Kiến trúc và hàm mất mát | GESF, cổng α, Logit Adjustment | Kiến trúc chuẩn (encoder + đầu phân loại), cross-entropy chuẩn; LA chỉ dùng như phép chẩn đoán (mục 6.4) |
| Hệ so sánh | B1–B5 | Ba bước: baseline, zero-shot, sequential, kèm đối chứng (mục 7.4) |
| Số lượt huấn luyện PLM | khoảng 200 | khoảng 65 (lõi), khoảng 125 nếu làm hết mở rộng (mục 9) |
| Tái sử dụng |  | Tiền xử lý, khử trùng lặp, seed, định dạng CSV, hash dữ liệu, bootstrap, cơ chế phối hợp nhóm |

## 2. Mục tiêu, câu hỏi nghiên cứu và giả thuyết

**Mục tiêu tổng quát:** đo lường và giải thích khả năng chuyển giao hai chiều giữa ngôn ngữ mạng xã hội (UIT-VSMEC) và ngôn ngữ học đường (NEU-ESC) với các PLM tiếng Việt.

**Mục tiêu cụ thể:** (1) thiết lập baseline đáng tin cậy trên từng miền; (2) định lượng độ lệch phân phối giữa hai miền bằng cả hiệu năng và các chỉ số nội tại; (3) đánh giá sequential transfer hai chiều và xác định chiều/thứ tự tối ưu; (4) đưa ra khuyến nghị thực hành kèm mã nguồn tái lập.

| Mã | Câu hỏi nghiên cứu | Giả thuyết (thăm dò, chưa có cơ sở chắc chắn) |
| --- | --- | --- |
| RQ1 | PhoBERT và ViSoBERT mạnh yếu thế nào trên từng miền? | H1: ViSoBERT ≥ PhoBERT trên VSMEC (tiền huấn luyện trên văn bản mạng xã hội); trên NEU-ESC hai mô hình ngang nhau hoặc PhoBERT nhỉnh hơn (văn phong trang trọng hơn). |
| RQ2 | Test thẳng sang miền kia thì suy giảm bao nhiêu, có bất đối xứng không, và do đâu (nhãn, từ vựng, văn phong)? | H2: Suy giảm lớn ở cả hai chiều; một phần đáng kể do lệch tiên nghiệm nhãn và ánh xạ nhãn 7 → 3, không chỉ do lệch ngôn ngữ. |
| RQ3 | Sequential transfer có cải thiện so với baseline không? | H3: Có nhưng bất đối xứng: NEU → VSMEC (trung gian lớn, đích nhỏ) cải thiện rõ hơn VSMEC → NEU (trung gian nhỏ, đích lớn), nơi lợi ích có thể bằng 0 hoặc âm. |
| RQ4 | Thứ tự nào là tối ưu? | H4: Miền học sau cùng quyết định hiệu năng trên đích; thứ tự tốt nhất cho một đích khác với thứ tự tốt nhất cho trung bình hai miền do hiện tượng quên. |
| RQ5 (mở rộng) | Lợi ích có tăng khi dữ liệu đích ít (10%)? | H5: Có, vì baseline yếu đi nhiều hơn khi dữ liệu đích giảm. |

## 3. Dữ liệu, không gian nhãn và tiền xử lý

### 3.1. Hai miền

| Tiêu chí | UIT-VSMEC | NEU-ESC |
| --- | --- | --- |
| Miền ngôn ngữ | Mạng xã hội | Học đường |
| Nhãn | 7 cảm xúc: Enjoyment, Sadness, Fear, Anger, Disgust, Surprise, Other | 4 nhãn gốc (neutral, positive, negative, toxic); gộp toxic vào negative thành 3 nhãn |
| Quy mô | khoảng 6,9 nghìn câu | 32.966 mẫu |
| Độ dài trung bình | 14,0 từ (p95 khoảng 32) | 25,1 từ (p95 là 83, dài nhất 1.578) |
| Đặc điểm nhãn | 7 lớp không cân bằng | neutral khoảng 69%, positive khoảng 12,6%, negative khoảng 18,3% |
| Vấn đề đã biết | 6 câu trùng | Không trùng lặp |

Các số liệu lấy từ kế hoạch cũ; **phải tính lại từ số đếm thực tế của từng split** trước khi đưa vào báo cáo.

### 3.2. Các góc nhìn nhãn (label view) và ánh xạ

Zero-shot không thể dự đoán nhãn nằm ngoài không gian đã huấn luyện, nên mọi so sánh liên miền được thực hiện trên **không gian 3 nhãn chung**:

- **V7:** VSMEC nguyên bản 7 lớp (dùng khi VSMEC là đích có huấn luyện).
- **V3-VSMEC:** VSMEC ánh xạ sang 3 lớp: Enjoyment → Positive; Sadness, Fear, Anger, Disgust → Negative; Surprise, Other → Neutral.
- **V3-NEU:** NEU-ESC 3 lớp (toxic gộp vào negative).
- **V4-NEU (tuỳ chọn):** giữ toxic thành lớp riêng, chỉ chạy nếu còn thời gian.

Mô hình 7 lớp được đưa về 3 lớp bằng cách **cộng xác suất theo nhóm** (cấu hình chính) và, để so sánh, lấy argmax rồi ánh xạ (cấu hình phụ). Mô hình NEU (3 lớp) được chấm trên nhãn VSMEC đã ánh xạ.

Ánh xạ là nguồn nhiễu vì Surprise có thể tích cực hoặc tiêu cực và Other không mặc nhiên trung lập. Hai biện pháp: (a) **VSMEC-strict:** báo cáo thêm trên tập con chỉ gồm Enjoyment và bốn cảm xúc tiêu cực (bỏ Surprise, Other); (b) **kiểm chứng thủ công** khoảng 100 câu Other và Surprise (chia đều), hai người gán cực tính độc lập, báo cáo tỉ lệ từng cực và Cohen's kappa. Quy tắc ánh xạ được chốt và đóng băng ở cuối tuần 2.

### 3.3. Tiền xử lý thống nhất

- Chuẩn hoá Unicode NFC, thu gọn ký tự lặp, giữ emoji; các bước còn lại (hạ chữ thường, v.v.) theo model card của từng backbone và áp dụng đồng nhất ở mọi thí nghiệm. PhoBERT cần tách từ bằng VnCoreNLP; ViSoBERT dùng tokenizer gốc.
- Khử trùng lặp trong từng split; câu trùng chéo split loại khỏi train.
- **Kiểm tra rò rỉ giữa hai bộ dữ liệu:** câu của VSMEC trùng với NEU-ESC (và ngược lại) bị loại khỏi tập **test** của cả hai miền, để mọi mô hình được chấm trên cùng tập test đã làm sạch; báo cáo số lượng bị loại.
- max\_length tính theo token: đo p95 và p99 của từng tokenizer trên NEU-ESC, dùng 256 (mức tối đa của PhoBERT), kèm dynamic padding và group\_by\_length; báo cáo tỉ lệ mẫu bị cắt.
- Chia dữ liệu: dùng split chính thức nếu có; nếu không, chia phân tầng 80/10/10 với seed cố định và lưu file chỉ mục. Test chỉ dùng để báo cáo. Lưu hash file đã xử lý và phiên bản mã tiền xử lý.
- Câu ≤ 2 từ giữ nguyên ở cấu hình chính; chạy cấu hình lọc để đo độ nhạy nếu còn thời gian.

## 4. Thiết lập chung cho mọi thí nghiệm

| Mục | Quy định |
| --- | --- |
| Backbone chính | PhoBERT (vinai/phobert-base-v2) và ViSoBERT (uitnlp/visobert); kiểm tra lại tên trên Hugging Face. Mở rộng: PhoBERT-large nếu còn GPU |
| Đầu phân loại | AutoModel encoder + Dropout + Linear dùng chung trong src/plm/model_ckpt.py, giữ tương thích checkpoint Bước 1 |
| Hàm mất mát | Cross-entropy chuẩn ở mọi cấu hình chính; class-weighted CE làm kiểm tra độ nhạy (NEU-ESC lệch 69% neutral) |
| Siêu tham số | Bước 1 dò LR {2e-5, 3e-5, 5e-5}; Bước 3 dò {1e-5, 2e-5, 3e-5}, chỉ trên dev. Batch hiệu dụng 32, tối đa 10 epoch, macro-F1 dev, patience 5, warmup 6%, weight decay 0,01 theo seq-v1. Sequential dùng bf16 CUDA strict; không fallback âm thầm. Cấu hình baseline thật cần đối chiếu log trước kết luận tương đương. |
| Seed | 5 seed cố định cho mọi cấu hình lõi; 3 seed cho cấu hình mở rộng |
| Chọn mô hình | Chỉ dùng dev của miền **đang được huấn luyện**. Ở Bước 2 tuyệt đối không dùng dữ liệu (kể cả dev) của miền đích |
| Độ đo chính | macro-F1; phụ: weighted-F1 (thường dùng trong các công bố về VSMEC), accuracy, F1 từng lớp, ma trận nhầm lẫn |

**Đặt tên model:** `{Backbone}_{Giai đoạn huấn luyện}`, ví dụ `PhoBERT_VSMEC`, `ViSoBERT_NEU`; sequential ghi `PhoBERT_VSMEC>NEU` (giai đoạn 1 > giai đoạn 2). Zero-shot không phải model mới nên ghi ở cột `eval_domain` và `mode`.

**File kết quả (CSV):** run\_id, backbone, stage1, stage2, eval\_domain, label\_view, mode (base / zs / seq), fraction, seed, macro\_f1, weighted\_f1, accuracy, f1\_per\_class, ckpt\_path.

## 5. Bước 1: Baseline

Train độc lập PhoBERT và ViSoBERT trên từng miền, ghi nhận F1-score làm mốc cho các bước sau.

| ID | Việc | Sản phẩm |
| --- | --- | --- |
| 1.1 | Train trên VSMEC (V7), 2 backbone × 5 seed. Đánh giá trên test VSMEC: macro-F1, weighted-F1, accuracy, F1 từng lớp. Tính thêm V3-VSMEC bằng ánh xạ (không cần train lại) | Bảng baseline VSMEC (V7 và V3) |
| 1.2 | Train trên NEU-ESC (V3-NEU), 2 backbone × 5 seed; đánh giá như trên | Bảng baseline NEU-ESC |
| 1.3 | Mốc sàn: majority-class và dự đoán ngẫu nhiên theo tiên nghiệm | Giá trị macro-F1 sàn cho mỗi bộ |
| 1.4 | Kiểm tra tính hợp lý: đối chiếu F1 trên VSMEC với số đã công bố của PhoBERT/ViSoBERT (tra bài báo gốc). Lệch nhiều thì rà lại pipeline trước khi đi tiếp | Ghi chú kiểm tra |
| 1.5 | Lưu checkpoint tốt nhất theo dev cho mỗi (backbone, miền, seed): 20 checkpoint | Dùng lại ở Bước 2 và 3 |

**Điều kiện qua bước:** kết quả tái lập được từ cấu hình, độ lệch chuẩn giữa các seed được báo cáo, baseline nằm trong vùng hợp lý so với công bố.

## 6. Bước 2: Zero-shot cross-domain transfer

### 6.1. Giao thức

- **Chiều A:** checkpoint VSMEC (V7) chấm trên test NEU-ESC, dự đoán 7 lớp được ánh xạ sang 3 lớp.
- **Chiều B:** checkpoint NEU-ESC chấm trên test VSMEC với nhãn đã ánh xạ 3 lớp (cả bản đầy đủ và VSMEC-strict).
- Không fine-tune, không chỉnh ngưỡng hay tiên nghiệm bằng dữ liệu đích. Tổng: 5 seed × 2 backbone × 2 chiều = 20 lượt suy luận (rất rẻ).
- **Tham chiếu trong miền, cùng không gian nhãn:** chiều A so với baseline NEU-3 (1.2); chiều B so với VSMEC-7 ánh xạ 3 lớp (1.1).

### 6.2. Đo độ lệch qua hiệu năng

- **Transfer gap** = F1\_trong\_miền − F1\_zero-shot; **retention** = F1\_zero-shot / F1\_trong\_miền. Báo cáo cùng mốc sàn của 1.3 để biết zero-shot còn hơn đoán bao nhiêu.
- **Bất đối xứng:** so sánh retention giữa hai chiều và giữa hai backbone.
- **Lệch dự đoán:** phân bố nhãn dự đoán so với nhãn thật (JS divergence); ma trận 7 × 3 (cảm xúc gốc của VSMEC → nhãn quan điểm mà mô hình NEU dự đoán) để thấy cảm xúc nào bị hiểu sai.

### 6.3. Đo độ lệch nội tại (không cần mô hình tinh chỉnh)

| Nhóm | Chỉ số | Cách đo |
| --- | --- | --- |
| Bề mặt | Độ dài, tỉ lệ OOV và độ trùng từ vựng, emoji, ký tự lặp, từ không dấu hoặc viết tắt | Thống kê trên cả hai tập; Jaccard top-k từ và KL unigram có làm trơn |
| Nhãn | Khác biệt tiên nghiệm nhãn | JS divergence trong không gian 3 lớp |
| Biểu diễn | Proxy A-distance (PAD) | Huấn luyện bộ phân loại tuyến tính phân biệt miền trên embedding mean-pooling, cân bằng kích thước hai miền; PAD = 2(1 − 2ε). Đo với encoder gốc và encoder đã fine-tune; vẽ t-SNE/UMAP tô theo miền và theo nhãn |

### 6.4. Phân tách nguyên nhân suy giảm (chỉ suy luận)

- **Oracle prior correction:** áp Logit Adjustment hậu nghiệm bằng tiên nghiệm đã biết của đích. Đây chỉ là chẩn đoán, không phải kết quả chính; phần F1 phục hồi cho biết bao nhiêu suy giảm đến từ lệch tiên nghiệm nhãn.
- **Độ nhạy ánh xạ:** so sánh cộng xác suất, argmax rồi ánh xạ, và VSMEC-strict.
- **Theo nhóm câu:** độ dài, có emoji, viết tắt, từ vựng học đường.
- Kết quả cuối là bảng quy suy giảm cho ba nhóm: (i) quy ước nhãn và ánh xạ, (ii) lệch tiên nghiệm, (iii) lệch ngôn ngữ còn lại.

## 7. Bước 3: Sequential transfer (intermediate fine-tuning)

### 7.1. Giao thức chính

- **S-A:** VSMEC → NEU-ESC. **S-B:** NEU-ESC → VSMEC. Hai chiều có trọng số như nhau.
- **Giai đoạn 1:** dùng lại checkpoint Bước 1 cùng seed, không train lại.
- **Giai đoạn 2:** giữ encoder, bỏ đầu phân loại cũ, khởi tạo đầu mới ngẫu nhiên (seed cố định), fine-tune toàn bộ trên miền đích (NEU-3 hoặc VSMEC-7), chọn mô hình theo dev đích, đánh giá trên test đích. Learning rate giai đoạn 2 dò {1e-5, 2e-5, 3e-5} trên dev đích với cùng ngân sách cho mọi cấu hình.
- 2 backbone × 2 chiều × 5 seed = 20 lượt. So sánh **có cặp** với baseline cùng seed trên cùng đích; ở đích VSMEC báo cáo cả V7 và V3.

### 7.2. Chỉ số

- **Gain** = F1\_sequential − F1\_baseline (macro-F1 chính; thêm weighted-F1 và F1 từng lớp, nhất là lớp thiểu số và neutral), kèm khoảng tin cậy.
- **Forgetting miền nguồn:** vì đầu cũ bị thay nên đo bằng **linear probe**: huấn luyện lại đầu tuyến tính trên encoder đóng băng bằng train miền nguồn, so F1 test nguồn sau giai đoạn 1 với sau giai đoạn 2. Biến thể V1 cho phép đo trực tiếp.
- **Độ dịch chuyển biểu diễn:** CKA hoặc khoảng cách embedding giữa encoder trước và sau giai đoạn 2.

### 7.3. Thế nào là thứ tự tối ưu

Chỉ có hai thứ tự nên được xét bằng thực nghiệm, theo hai tiêu chí tách bạch:

- **(a) Tối ưu cho một đích:** thứ tự có đích ở cuối so với baseline của đích đó (đã nằm trong 7.1).
- **(b) Tối ưu cho cả hai miền:** so VSMEC → NEU, NEU → VSMEC và huấn luyện đồng thời (V2) theo trung bình macro-F1 trên hai test, dùng biến thể V1 (đầu 3 lớp chung) để đo cả hai miền trực tiếp.

Chiều tối ưu có thể khác nhau giữa hai backbone; báo cáo trung thực thay vì gộp.

### 7.4. Biến thể và đối chứng

| ID | Biến thể | Mục đích | Ưu tiên |
| --- | --- | --- | --- |
| C0 | Self-sequential: fine-tune lần hai trên **cùng** miền đích với đầu mới | Loại trừ giải thích lợi ích chỉ do huấn luyện thêm | Cao (rẻ trên VSMEC) |
| V1 | Shared-3: giai đoạn 1 trên VSMEC-3 (nhãn đã ánh xạ) rồi giai đoạn 2 trên NEU-3 giữ nguyên đầu 3 lớp, và chiều ngược lại | Nhãn cảm xúc mịn (7) có giá trị hơn nhãn thô (3)? Giữ đầu so với thay đầu; đo forgetting trực tiếp | Cao |
| V2 | Joint: huấn luyện đồng thời hai miền (encoder chung, hai đầu, trộn batch) | Mốc tham chiếu không có thứ tự cho RQ4 | Trung bình |
| V3 | Ổn định giai đoạn 2: learning rate thấp hơn, layer-wise LR decay hoặc linear-probe rồi fine-tune | Giảm quên và méo biểu diễn | Mở rộng |
| V4 | Đích ít dữ liệu: 10% phân tầng, dev thu nhỏ cùng tỉ lệ | RQ5 | Mở rộng (3 seed) |
| V5 | Class-weighted CE hoặc Logit Adjustment ở giai đoạn 2 | Kiểm tra lợi ích không chỉ do cân bằng nhãn | Mở rộng |
| V6 | PhoBERT-large | Kiểm tra theo quy mô mô hình | Mở rộng |

## 8. Thống kê, phân tích kết quả và tái lập

- **So sánh chính (chốt trước khi chạy Bước 3):** Sequential so với Baseline trên đích, theo từng chiều và từng backbone, tức 4 so sánh: VSMEC → NEU và NEU → VSMEC, mỗi chiều với PhoBERT và ViSoBERT. Chỉ số: macro-F1 trên test đích (V3-NEU hoặc V7). Hiệu chỉnh đa so sánh bằng Holm trên 4 so sánh này; mọi so sánh khác là thăm dò.
- **Kiểm định:** paired bootstrap hai tầng (lấy lại mẫu test và seed) hoặc kiểm định hoán vị, mức 0,05. Không dùng Wilcoxon signed-rank làm kiểm định chính vì với 5 seed, p hai phía nhỏ nhất chỉ là 0,0625.
- **Lưu ý độ nhiễu:** test VSMEC chỉ khoảng 700 câu nên khoảng tin cậy rộng; chênh lệch macro-F1 vài điểm có thể không phân biệt được với nhiễu. Báo cáo mean ± std, khoảng tin cậy 95% và diễn giải thận trọng.
- **Phân tích lỗi định tính:** F1 từng lớp và ma trận nhầm lẫn; ma trận 7 × 3 của VSMEC; nhóm câu (dài/ngắn, emoji, viết tắt, mỉa mai, từ vựng học đường); ví dụ câu sequential sửa đúng và câu làm sai thêm; t-SNE/UMAP trước và sau giai đoạn 2.
- **Tái lập:** lưu YAML cấu hình, seed, phiên bản thư viện, commit mã nguồn, hash dữ liệu, checkpoint tốt nhất và file CSV kết quả theo định dạng ở mục 4.

## 9. Ngân sách chạy và tài nguyên

| Nhóm | Cấu hình | Số lượt |
| --- | --- | --- |
| Dò learning rate Bước 1 | 3 LR × 2 backbone × 2 miền (1 seed) | 12 |
| Bước 1 | 2 backbone × 2 miền × 5 seed | 20 |
| Bước 2 | 20 lượt suy luận, chỉ trích xuất dự đoán và embedding | 0 (không train) |
| Dò learning rate giai đoạn 2 | 3 LR × 2 backbone × 2 chiều (1 seed) | 12 |
| Bước 3 lõi | 2 backbone × 2 chiều × 5 seed | 20 |
| **Tổng lõi** |  | **khoảng 65** |
| C0 (self-sequential) | 2 backbone × 2 miền × 3 seed | 12 |
| V1 (shared-3) | 6 lượt giai đoạn 1 + 12 lượt giai đoạn 2 | 18 |
| V2 (joint) | 2 backbone × 3 seed | 6 |
| V4 (đích 10%) | 2 chiều × 2 backbone × 3 seed × (baseline + sequential) | 24 |
| **Tổng nếu làm hết mở rộng** |  | **khoảng 125** |

Ước lượng thô: một lượt trên VSMEC mất vài phút, một lượt trên NEU-ESC khoảng 10–30 phút trên A100; **phải đo thực tế ở tuần 3**. Thứ tự cắt giảm nếu vượt quỹ thời gian: bỏ V6 (PhoBERT-large), V3, V5, rồi V4, rồi giảm seed của các biến thể mở rộng. Tài nguyên: Kaggle GPU (lưu ý hạn mức không cố định, phiên tối đa 12 giờ) và Colab A100; PyTorch, Hugging Face Transformers, scikit-learn, VnCoreNLP (cần Java), W&B hoặc MLflow. Lưu checkpoint thường xuyên.

## 10. Tiến độ 14 tuần

| Tuần | Công việc | Mốc |
| --- | --- | --- |
| 1–2 | Tiền xử lý, khử trùng lặp và kiểm tra rò rỉ chéo, chia tập, hash; gán nhãn thủ công Other/Surprise; đo độ lệch nội tại (6.3) | Cuối tuần 2: dữ liệu và quy tắc ánh xạ đóng băng |
| 3 | Chạy thử toàn chuỗi 3 bước (1 seed, 1 backbone), đo thời gian thực, chốt 4 so sánh chính | Go/no-go cấu hình |
| 4–5 | Bước 1 (kèm dò LR) | Bảng baseline và 20 checkpoint |
| 6 | Bước 2 và phân tích độ lệch (6.2–6.4) | Báo cáo distribution shift |
| 7–8 | Bước 3 lõi cả hai chiều | Bảng chính |
| 9 | C0 và V1 | Go/no-go cho phần mở rộng |
| 10–11 | V2, V4 (nếu còn); phân tích lỗi, thống kê, hình | Bộ kết quả cuối |
| 12–13 | Hoàn thiện báo cáo (viết dần từ tuần 4) | Bản nháp báo cáo |
| 14 | Vùng đệm, chuẩn bị bảo vệ |  |

## 11. Phân công

| Thành viên | Nhiệm vụ | Sản phẩm |
| --- | --- | --- |
| Nhân sự 1: Dữ liệu | Pipeline tiền xử lý, khử trùng lặp, kiểm tra rò rỉ, chia tập, hash; kiểm chứng ánh xạ Other/Surprise; đo độ lệch nội tại (6.3); script đánh giá chung | Dữ liệu chuẩn, thống kê cực tính, báo cáo độ lệch nội tại |
| Nhân sự 2: PLM cơ bản | Bước 1 (dò LR, train, lưu checkpoint); chạy C0 và V1 | Bảng baseline, 20 checkpoint |
| Chủ nhiệm | Pipeline Bước 2 và 3, chẩn đoán nguyên nhân (6.4), thống kê, phân tích lỗi, rà soát mã chéo, biên tập báo cáo | Bảng sequential, kết luận chính |
| Phần học tập (ngoài lõi) | Thành viên mới chạy TF-IDF + LR/SVM và BiLSTM theo cùng giao thức Bước 1–2 để thấy khác biệt giữa PLM và mô hình cổ điển | Phụ lục, không dùng cho kết luận chính |

**Phối hợp:** một repository, một bộ tiền xử lý, một script đánh giá, họp ngắn hàng tuần, đóng băng dữ liệu và cấu hình đánh giá ở tuần 3.

## 12. Rủi ro và phương án xử lý

| Rủi ro | Biện pháp |
| --- | --- |
| Ánh xạ 7 → 3 nhiễu làm méo kết quả liên miền | Kiểm chứng thủ công Other/Surprise, báo cáo VSMEC-strict và độ nhạy ánh xạ |
| Test VSMEC nhỏ, khoảng tin cậy rộng | 5 seed, bootstrap, diễn giải thận trọng, không kết luận từ chênh lệch nhỏ |
| Lợi ích sequential chỉ do huấn luyện thêm | Đối chứng C0 |
| Rò rỉ dữ liệu giữa hai bộ | Kiểm tra trùng chéo, loại khỏi test, lưu hash |
| Chuyển giao âm hoặc không đổi | Đây là kết quả hợp lệ; phân tích nguyên nhân bằng 6.4, V1 và V3 |
| Lỗi pipeline (tách từ PhoBERT, độ dài token) | Chạy thử tuần 3, kiểm tra chéo mã, đối chiếu số đã công bố ở 1.4 |
| Số liệu NEU-ESC chưa được xác nhận | Tính lại từ số đếm thực tế từng split |
| Giới hạn GPU, phiên Kaggle hoặc Colab ngắt | Lưu checkpoint, bf16, gradient accumulation, thứ tự cắt giảm ở mục 9 |

## 13. Sản phẩm đầu ra và tiêu chí thành công

**Sản phẩm:** mã nguồn, cấu hình và checkpoint chính; dữ liệu đã làm sạch kèm danh sách mẫu bị loại; bảng kết quả ba bước; báo cáo độ lệch phân phối; báo cáo nghiệm thu; bài báo hội thảo hoặc tạp chí trong nước (định hướng); bản trình bày bảo vệ.

**Tiêu chí thành công:**

- Hoàn thành ba bước lõi với 5 seed và tái lập được từ mã nguồn và cấu hình.
- Baseline nằm trong vùng hợp lý so với kết quả đã công bố.
- Định lượng được độ lệch ở cả hai chiều và phân tách được nguyên nhân (mục 6.4).
- Có kết luận rõ cho 4 so sánh chính; kết quả dương, bằng 0 hay âm đều được chấp nhận nếu được phân tích.
- Trả lời được câu hỏi chiều và thứ tự tối ưu kèm khoảng tin cậy.

## 14. Điểm cần xác nhận

- Cụm "model lớn" được hiểu là PLM cỡ base (PhoBERT-base-v2, ViSoBERT); PhoBERT-large là mở rộng.
- Quy tắc ánh xạ 7 → 3 (mục 3.2) và việc chỉ báo cáo liên miền trong không gian 3 nhãn.
- Có giữ UIT-VSFC làm miền thứ ba hay bỏ hẳn.
- Số liệu và split của NEU-ESC cần tính lại từ dữ liệu thật.
