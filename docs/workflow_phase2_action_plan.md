# Lộ trình triển khai nghiên cứu giai đoạn 2

**UIT-VSMEC ↔ NEU-ESC | Bản hướng dẫn thực hiện từng bước**  
**Trạng thái:** Đề xuất triển khai, chưa xác nhận tài nguyên thực tế.  
**Tài liệu tham chiếu:** `Ke_hoach_giai_doan_tiep_theo_tinh_gon (1).docx` và `workflow_phase2_resource_transfer.md` (26/11/2026).

> **Bắt đầu ở Task 01.** Chưa cần chạy GPU, viết LoRA hay đọc toàn bộ đặc tả kỹ thuật. Chỉ chuyển sang công việc tiếp theo khi đầu ra của công việc hiện tại đã được kiểm tra.

## 1. Nghiên cứu này nhằm trả lời điều gì?

- Khi giảm dữ liệu đích xuống 10%, 25%, 50%, chuyển giao học tập có còn hữu ích so với huấn luyện trực tiếp không?
- LLRD và LoRA có giúp giảm chuyển giao âm và độ dao động giữa các seed không?
- ViSoBERT có sự đánh đổi hiệu năng – chi phí như thế nào so với XLM-R và LLM ở mức 10%?

Hai chiều nghiên cứu:

| Chiều | Nguồn (100%) | Đích | Seq − DFT cũ ở 100% |
|---|---|---|---:|
| A (SA) | UIT-VSMEC | NEU-ESC | −0,0441 |
| B (SB) | NEU-ESC | UIT-VSMEC | +0,0635 |

**Các phương pháp:** DFT (tên cũ REF) là huấn luyện trực tiếp từ mô hình tiền huấn luyện trên dữ liệu đích; Seq là học từ checkpoint nguồn rồi thích nghi đích; C0 là đối chứng huấn luyện hai giai đoạn trên miền đích (cần sửa giao thức low-resource, xem Task 04). LLRD là tốc độ học giảm dần về các tầng thấp; LoRA là tinh chỉnh adapter và đầu phân loại thay vì toàn bộ encoder.

## 2. Theo dõi tiến độ

| Task | Công việc | Điều kiện bắt đầu | Trạng thái |
|---|---|---|---|
| 01 | Kiểm kê tài nguyên cũ | Không | [ ] |
| 02 | Kiểm tra dữ liệu và giao thức cũ | 01 | [ ] |
| 03 | Tạo tập con dữ liệu | 02 | [ ] |
| 04 | Chốt giao thức thực nghiệm | 02–03 | [ ] |
| 05 | Chuẩn bị code DFT, C0, Seq | 04 | [ ] |
| 06 | Chạy thử GPU | 05 | [ ] |
| 07 | Dò learning rate | 06 | [ ] |
| 08 | Thực nghiệm chính 10% | 07 | [ ] |
| 09 | Mở rộng 25%, 50% | 08 | [ ] |
| 10 | Thực nghiệm LLRD, LoRA | 07; cần Seq tương ứng để kết luận | [ ] |
| 11 | Thực nghiệm XLM-R, LLM | 03–04; code phù hợp | [ ] |
| 12 | Phân tích và báo cáo | 08–11 | [ ] |

Luồng ưu tiên: **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12**. Task 10 và 11 có thể chạy song song khi đủ điều kiện kỹ thuật. Không cần hoàn tất toàn bộ code của Task 10–11 để bắt đầu Task 06.

## 3. Hướng dẫn từng Task

### Task 01 — Kiểm kê tài nguyên cũ (làm ngay)

**Mục tiêu:** Biết mình đang có gì và thiếu gì, tránh làm lại giai đoạn 1.

**Đầu vào:** Repository, dữ liệu, checkpoint và kết quả giai đoạn 1. Workflow cũ ghi nhánh `phase2-resource-transfer` tách từ `seq-core` tại `62df4cd`; cần kiểm tra thực tế, không mặc định đã tồn tại.

**Việc cần làm:**
- [ ] Tìm train/dev/test của UIT-VSMEC và NEU-ESC.
- [ ] Tìm 10 checkpoint nguồn ViSoBERT (2 miền × seed 0–4), tokenizer và metadata/hash nếu có.
- [ ] Tìm code, cấu hình và kết quả DFT/REF, C0, Seq, ZS cũ; tìm dự đoán từng mẫu test của ViSoBERT và PhoBERT ở 100%.
- [ ] Ghi đường dẫn, trạng thái có/thiếu và lỗi phát hiện; không sửa file gốc.

**Đầu ra:** `reports/phase2/inventory.md` (đường dẫn đề xuất).

Mẫu bảng điền ngay:

| Tài nguyên | Đường dẫn thực tế | Có/thiếu | Ghi chú |
|---|---|---|---|
| UIT-VSMEC train/dev/test | | | |
| NEU-ESC train/dev/test | | | |
| Checkpoint nguồn UIT-VSMEC | | | |
| Checkpoint nguồn NEU-ESC | | | |
| Code DFT/C0/Seq | | | |
| Cấu hình giai đoạn 1 | | | |
| Kết quả và dự đoán 100% | | | |

**Hoàn thành khi:** Xác định rõ tài nguyên nào dùng lại được và tài nguyên nào cần bổ sung. Không cần GPU. **Tiếp theo: Task 02.**

### Task 02 — Kiểm tra dữ liệu và giao thức cũ

**Mục tiêu:** Xác nhận dữ liệu và mốc 100% đáng tin cậy.

**Đầu vào:** Danh sách Task 01.

- [ ] Kiểm tra số câu, ID, nhãn, trùng lặp và hash giữa train/dev/test; đối chiếu MANIFEST với file thật, đặc biệt `polarity_long.csv` và `polarity_nhat.csv` đang được báo cáo có hash giống nhau dù khác kích thước.
- [ ] Kiểm tra chuẩn hóa NFC, ký tự lặp, emoji; ViSoBERT/XLM-R phải dùng văn bản chưa tách từ, tokenizer đúng, `max_length=256`. Đối chiếu trực tiếp nội dung/hash đầu vào và tỷ lệ dấu `_`, ghi tỷ lệ cắt cụt.
- [ ] Đối chiếu cách chọn checkpoint, dev, tiền xử lý, nhãn và seed của kết quả 100% cũ với giao thức dự kiến.
- [ ] Sửa notebook Probe: PRIOR từ train thay vì test; tokenizer đúng checkpoint, không fallback ngầm. Có thể xử lý song song nhưng phải hoàn tất trước khi dùng kết quả probe trong báo cáo.

**Đầu ra:** `reports/phase2/data_audit.md`, `manifest_audit.json` và ghi chú khả năng tái sử dụng mốc 100%.

**Hoàn thành khi:** Không còn lỗi dữ liệu chưa xử lý ảnh hưởng đến việc tạo tập con; lỗi manifest được ghi nhận và có phương án được duyệt. Không tự ý thay đổi dữ liệu đóng băng.

### Task 03 — Tạo tập con dữ liệu

**Mục tiêu:** Có các tập train nhỏ, công bằng và tái lập được.

| Mức | UIT-VSMEC (dự kiến) | NEU-ESC (dự kiến) |
|---|---:|---:|
| 5% (tùy chọn) | 277 | 1.152 |
| 10% | 554 | 2.305 |
| 25% | 1.386 | 5.762 |
| 50% | 2.771 | 11.524 |
| 100% | 5.542 | 23.048 |

- [ ] Dùng dữ liệu train đã xác nhận ở Task 02, phân tầng theo nhãn gốc sử dụng trong giao thức, cho seed 0–4.
- [ ] Với mỗi seed, tạo thứ tự ID cố định trong từng lớp; phân bổ số mẫu theo tỷ lệ với ràng buộc số mẫu từng lớp không giảm khi tăng mức. Kiểm thử thực tế 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%; không chỉ dựa vào largest-remainder độc lập.
- [ ] Lưu ID và hash vào `data/subsets_phase2/`, không ghi đè `data/splits/`.
- [ ] Kiểm tra không giao ID hoặc hash văn bản với dev/test; lệch tỷ lệ từng lớp tối đa 1 điểm phần trăm ở 10/25/50%; cùng mức và seed thì mọi phương pháp dùng cùng tập con.
- [ ] Xuất bảng đếm từng lớp mức 5% của UIT-VSMEC cho cả 5 seed; nếu bất kỳ lớp nào dưới 5 mẫu thì bỏ mức 5% cho V7. Quyết định chạy 5% của NEU-ESC riêng ở Task 04.

**Đầu ra:** File ID, MANIFEST mới, báo cáo kiểm tra tập con và `vsmec_5pct_class_counts.csv`.

**Hoàn thành khi:** Tất cả kiểm tra bắt buộc đạt; quy mô và phân phối thực tế được ghi lại. Dev/test giữ nguyên, không rút theo tỷ lệ train.

### Task 04 — Chốt giao thức thực nghiệm (cổng bắt buộc trước train)

**Mục tiêu:** Tránh chạy hàng loạt rồi phải chạy lại vì thiết kế không nhất quán.

| Quyết định | Đề xuất để nhóm xác nhận |
|---|---|
| C0 low-resource | Cả hai pha C0 chỉ được dùng cùng tập train đích ở mức và seed tương ứng; không dùng checkpoint đã thấy 100% đích. Chốt khởi tạo head, lịch train và ngân sách cập nhật đối chứng. |
| Dev và checkpoint | Ưu tiên dev gốc cố định; UIT-VSMEC ít nhất 172 câu. Nếu tương thích mốc cũ, giữ tiêu chí cũ; nếu chọn Macro-F1 5 lớp cho UIT-VSMEC và V3-NEU cho NEU-ESC thì áp dụng đồng nhất và xử lý khác biệt mốc cũ. |
| Mốc 100% | Chỉ coi là điểm cùng giao thức nếu kiểm tra tương thích; nếu khác, trình bày riêng hoặc chạy lại cấu hình cần thiết sau khi nhóm duyệt. |
| Seed và thống kê | DFT/Seq/LLRD/LoRA: 0–4; C0: 0–2. So sánh Seq−C0 trên seed chung. Chốt cách tính p-value bootstrap hợp lệ và Holm m=2 trước khi xem kết quả test. |
| Phạm vi | ViSoBERT chính; cả LLRD và LoRA ở 10% hai chiều và 100% chiều A theo ngân sách cơ sở. Các mức 5%, Pha 2 ở 25%, mDeBERTa là tùy chọn. |
| Tên gọi | DFT trong báo cáo, ánh xạ REF cũ rõ ràng. |

**Lưu ý quan trọng:** Sửa C0 là đề xuất thay đổi khoa học so với workflow cũ, không phải quyết định đã được nhóm phê duyệt. C0 cần ngân sách hai pha huấn luyện có thể kiểm chứng; không tuyên bố đã cân bằng số bước nếu chưa đo. Với các seed chung, nếu tái sử dụng DFT cùng tập con làm pha đầu C0 thì phải xác nhận phù hợp giao thức.

**Đầu ra:** `configs/phase2.yaml` (`protocol_version: phase2-v1`) và `reports/phase2/decisions.md`. Giữ `configs/seq.yaml` và dữ liệu cũ không đổi.

**Hoàn thành khi:** Nhóm xác nhận các mục ảnh hưởng đến DFT/C0/Seq; lựa chọn LLM cụ thể có thể hoãn đến Task 11. Không dùng test để lựa chọn giao thức.

### Task 05 — Chuẩn bị code DFT, C0, Seq

**Mục tiêu:** Code thực nghiệm lõi chạy đúng trước khi thêm phương pháp mới.

- [ ] DFT khởi tạo từ ViSoBERT tiền huấn luyện, head đích mới; Seq dùng checkpoint nguồn cùng seed, head đích theo giao thức.
- [ ] C0 thực hiện đúng giao thức đã duyệt ở Task 04; ghi nguồn checkpoint và hash tập dữ liệu đã tiếp cận.
- [ ] Đọc tập con theo ID/seed, dev/test cố định; chọn best checkpoint chỉ theo dev.
- [ ] Lưu cấu hình, seed, hash, dự đoán test theo ID, xác suất từng lớp, thời gian, GPU, VRAM và số tham số. Tách tổng tham số (mọi tham số) và tham số trainable (`requires_grad=True`).
- [ ] Unit test cho tính lồng nhau, lọc ID, chống rò rỉ, tiêu chí dev, resume, đếm tham số và SKIP chỉ khi đầu ra hoàn chỉnh.

**Đầu ra:** Code/module mới hoặc phần mở rộng có kiểm thử, cấu hình và log pytest.

**Hoàn thành khi:** Test cũ và mới đều PASS. Chưa cần hoàn thiện LoRA, LLRD, XLM-R tại bước này.

### Task 06 — Chạy thử một thực nghiệm GPU

**Mục tiêu:** Xác nhận pipeline hoạt động end-to-end, đo chi phí trước khi chạy hàng loạt.

- [ ] Chạy thử DFT ViSoBERT, đích NEU-ESC 10%, seed 0 (cấu hình thử đề xuất; chỉ dùng dev).
- [ ] Kiểm tra dữ liệu thực tế, loss, checkpoint, metric, resume và log thời gian/VRAM. Kiểm tra pipeline C0/Seq bằng test hoặc smoke bổ sung nếu cần.
- [ ] Ước lượng GPU-giờ theo thời gian từng loại lượt chạy; không coi thời gian DFT đại diện chính xác cho mọi phương pháp.

**Đầu ra:** `reports/phase2/smoke_test.md` và log chạy.

**Hoàn thành khi:** Chạy thành công và kiểm tra được đầu ra; chưa xem test để chọn tham số.

### Task 07 — Dò learning rate

**Mục tiêu:** Khóa learning rate cho thực nghiệm lõi.

- [ ] DFT: 2 đích × 3 LR (2e-5, 3e-5, 5e-5) × seed 0 = 6 lượt.
- [ ] Seq: 2 chiều × 3 LR (1e-5, 2e-5, 3e-5) × seed 0 = 6 lượt.
- [ ] Chỉ dùng train 10% và dev. Chọn LR theo điểm dev, hòa thì chọn LR nhỏ hơn; mở rộng lưới nếu nghiệm ở biên theo quy tắc đã chốt.
- [ ] Khóa LR cho các mức lớn hơn; LR của C0 theo giao thức đối chứng đã duyệt.

**Đầu ra:** `results/phase2/selection_p1_lr.json` và bảng sweep.

**Hoàn thành khi:** Có LR cố định cho từng phương pháp/chiều; không dùng test để thay đổi lựa chọn.

### Task 08 — Thực nghiệm chính ở mức 10%

**Mục tiêu:** Hoàn tất hai so sánh chính của nghiên cứu.

- [ ] Với mỗi chiều, chạy DFT 5 seed, Seq 5 seed, C0 3 seed (26 cấu hình tổng cộng cho hai chiều).
- [ ] Kiểm tra file dự đoán test theo ID, xác suất hợp lệ (tổng gần 1 trong sai số số học), hash và log chi phí.
- [ ] Lập bảng Macro-F1, weighted-F1, F1 từng lớp và Gain sơ bộ; không đổi thiết kế dựa trên test.

**Đầu ra:** Kết quả và dự đoán 10%, bảng kiểm tra 26 cấu hình.

**Hoàn thành khi:** Đủ cấu hình hợp lệ; không nhất thiết Seq phải thắng DFT. Kết quả âm cũng là kết quả nghiên cứu.

### Task 09 — Mở rộng sang 25% và 50%

**Mục tiêu:** Xác định xu hướng chuyển giao khi lượng dữ liệu tăng.

- [ ] Lặp lại Task 08 ở 25% và 50% với LR đã khóa: 52 cấu hình.
- [ ] Nối với mốc 100% nếu tương thích theo Task 04; nếu không, ghi chú khác biệt hoặc dùng mốc đã chạy lại được duyệt.
- [ ] Chỉ thêm 5% nếu đủ điều kiện số mẫu và được nhóm duyệt; nếu thiếu GPU, ưu tiên giữ 10%, 25%.

**Đầu ra:** Bảng kết quả theo hai chiều, tỷ lệ và số câu tuyệt đối.

**Hoàn thành khi:** Đủ 78 cấu hình lõi (Task 08 + 09), hoặc có phạm vi cắt giảm được ghi nhận.

### Task 10 — Thử LLRD và LoRA trên Seq

**Mục tiêu:** Kiểm tra giảm chuyển giao âm và biến động giữa seed.

- [ ] Cài đặt và kiểm thử LLRD (hệ số 0,8/0,9) và LoRA (r=8/16, alpha=2r, dropout 0,1, query/value, head train đầy đủ). Kiểm tra tham số đóng băng không bị cập nhật.
- [ ] Sau Task 07, dò 16 cấu hình trên train 10%/dev seed 0: LLRD 4 lượt dùng LR Seq đã khóa; LoRA 12 lượt với LR 1e-4, 3e-4, 5e-4. Khóa cấu hình tốt nhất cho từng chiều/phương pháp.
- [ ] Chạy chính 20 cấu hình ở 10% hai chiều và 10 cấu hình ở 100% chiều A. Chỉ thêm 25% nếu được duyệt và còn GPU.
- [ ] So sánh với Seq cùng mức, seed, giao thức; nếu mốc Seq 100% không tương thích, chưa kết luận cải thiện trực tiếp.

**Đầu ra:** `selection_p2_hparams.json`, dự đoán, log chi phí, bảng Gain và độ lệch chuẩn.

**Hoàn thành khi:** Đủ 30 cấu hình chính theo phạm vi cơ sở; 16 lượt dò chỉ dùng dev. Không khẳng định giảm forgetting khi chưa đo trực tiếp; thiếu DFT+LLRD/LoRA nên không tách được điều chuẩn thuần.

### Task 11 — So sánh XLM-R và LLM

**Mục tiêu:** Có mốc tham chiếu hiệu năng – chi phí ở mức 10%.

- [ ] XLM-R-base: DFT 100% hai miền (6 lượt, dùng làm checkpoint nguồn), DFT 10% (6), Seq 10% (6), ZS hai chiều chỉ suy luận. Seed 0–2, cùng tập con và góc nhìn nhãn; chốt LR phù hợp trên dev trước khi chạy chính.
- [ ] Chọn LLM mở 7–9B hỗ trợ tiếng Việt, xác minh phiên bản và giấy phép tại thời điểm chạy. Ghi model, precision/quantization.
- [ ] LLM zero-shot/few-shot, không huấn luyện, greedy decoding, đầu ra trong tập nhãn đích; chỉnh prompt chỉ trên dev, 3 cách chọn ví dụ từ train 10%.
- [ ] Đo trên cùng môi trường GPU chuẩn: tổng/trainable params, VRAM, thời gian train, độ trễ batch 1 và batch lớn, throughput, GPU-giờ/1.000 câu. Ghi rõ batch, precision và điều kiện đo.

**Đầu ra:** Kết quả XLM-R/LLM, bảng đánh đổi hiệu năng – chi phí.

**Hoàn thành khi:** Đủ mốc so sánh được duyệt; ghi rõ nguy cơ dữ liệu công khai đã xuất hiện trong pretraining của LLM. mDeBERTa chỉ là tùy chọn.

### Task 12 — Phân tích kết quả và viết báo cáo

**Mục tiêu:** Trả lời câu hỏi nghiên cứu từ kết quả đã khóa, không chạy thêm chỉ để tìm kết quả đẹp.

- [ ] Tính Macro-F1, weighted-F1, F1 từng lớp; UIT-VSMEC: V7, Macro-F1 5 lớp (trung bình F1 của 5 lớp trên dự đoán V7), V3, V3-noSurprise, Strict V3; NEU-ESC: V3-NEU, V3-NEU-noToxic. Giữ đúng quy tắc ánh xạ của dự án.
- [ ] Paired bootstrap hai tầng (10.000 lần theo đề xuất workflow cũ), ghép seed và ID test chung, CI 95%. Holm m=2 cho Seq−DFT ở 10% hai chiều; các so sánh còn lại là thăm dò. Dùng quy tắc p-value đã duyệt.
- [ ] Vẽ Gain theo tỷ lệ và số mẫu: Seq−DFT, Seq−C0; xác định mức có lợi ích rõ nhất theo tiêu chí kế hoạch (Seq−C0 cao nhất với cận dưới CI > 0). Phân biệt đổi dấu điểm ước lượng với CI chứa 0; chỉ kết luận tại các mức quan sát được.
- [ ] Tính SD giữa seed và CI cho tỷ số SD của LLRD/LoRA so với Seq; nếu SD mẫu số bằng 0 thì ghi không xác định. Với 5 seed chỉ diễn giải thăm dò.
- [ ] Lập bảng đối chiếu nhãn/phân phối lớp và kiểm tra tính nhất quán của Gain trên các góc nhìn; ánh xạ Surprise/Other → Neutral là giả định, không phải sự thật đã xác nhận.
- [ ] Hoàn tất nhật ký từng cấu hình, bảng chi phí, hạn chế và báo cáo. Phân biệt phát hiện thực nghiệm với giải thích cơ chế giả thuyết.

**Đầu ra:** Bảng kết quả, biểu đồ Gain, báo cáo thống kê, bảng Dataset, bảng chi phí và bản nháp bài nghiên cứu.

**Hoàn thành khi:** Trả lời được các câu hỏi nghiên cứu, có bằng chứng và hạn chế rõ ràng, kể cả khi giả thuyết không được ủng hộ.

## 4. Ngân sách và thứ tự ưu tiên

| Hạng mục | Cấu hình/lượt theo dự toán gốc |
|---|---:|
| Task 07: dò LR | 12 |
| Task 08: lõi 10% | 26 |
| Task 09: lõi 25/50% | 52 |
| Task 10: LLRD/LoRA (dò + chạy chính) | 46 |
| Task 11: XLM-R | 18 |
| Tổng cơ sở | 154 |

154 là dự toán gốc, **không phải tổng lượt GPU đã khóa**: chưa tính smoke, mở rộng lưới, tuning XLM-R nếu cần, suy luận LLM, chạy lại mốc 100% và chi phí pha đầu C0 nếu không tái sử dụng được DFT. Phải cập nhật ngân sách sau Task 04 và 06. Tùy chọn: 5% tối đa +26 cấu hình lõi (nếu cả hai đích đủ điều kiện), LLRD/LoRA 25% +20, mDeBERTa +18.

Nếu thiếu GPU, cắt theo thứ tự kế hoạch gốc: 5% → LLRD/LoRA 25% → mDeBERTa → mức 50% → giảm seed C0 (chỉ sau khi duyệt và cập nhật phân tích ghép cặp). Bảo vệ 5 seed DFT/Seq tại 10%.

## 5. Quy tắc dùng tài liệu

- File này là checklist thực thi hằng ngày. Workflow cũ vẫn là tài liệu kỹ thuật tham chiếu, nhưng các điểm mâu thuẫn phải được sửa và xác nhận ở Task 04 trước khi dùng để chạy.
- Siêu tham số chung tham chiếu kế hoạch: batch hiệu dụng 32, warmup 10%, weight decay 0,01, tối đa 10 epoch, patience 2, max_length 256; LoRA có thể nâng 15 epoch nếu chưa hội tụ trên dev theo quy tắc đã chốt. Dùng bf16 nếu GPU hỗ trợ. Chi tiết triển khai lưu trong cấu hình và đặc tả kỹ thuật.
- Không sửa âm thầm dữ liệu/cấu hình cũ, không chọn tham số hoặc prompt bằng test, không mở rộng PhoBERT hay các thí nghiệm ngoài phạm vi.
- Sau mỗi Task, ghi đầu ra thực tế, lỗi còn tồn tại, quyết định cần duyệt và Task tiếp theo. Không đánh dấu hoàn thành nếu chưa kiểm tra.

**Các thay đổi đề xuất so với workflow cũ:** sửa C0 low-resource, kiểm tra tính tương thích mốc 100%, bổ sung ràng buộc lồng nhau, tách cách đếm tham số, sửa phụ thuộc LLRD vào LR Seq, tách đổi dấu khỏi CI chứa 0 và cập nhật cách hiểu ngân sách. Đây là đề xuất cần nhóm xác nhận, không phải kết quả kiểm tra repository.

---

**Việc duy nhất cần làm ngay:** Mở repository giai đoạn 1, điền bảng kiểm kê ở Task 01. Chưa chạy GPU.
