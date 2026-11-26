# Hướng dẫn triển khai Task 05 và Task 10

**Dự án:** Chuyển giao có kiểm soát giữa UIT-VSMEC và NEU-ESC — giai đoạn 2  
**Dành cho:** Hai thành viên phụ trách code thực nghiệm chính và cải tiến mô hình.  
**Tham chiếu:** `workflow_phase2_action_plan.md` (Task 04–10), `workflow_phase2_resource_transfer.md` (bước 0.4, 1.1, 2.1–2.3).

> Tình hình hiện tại: Task 01 và 02 đã hoàn thành theo thông tin của nhóm; người điều phối dự định hoàn thành Task 03–04. Hai thành viên có thể chuẩn bị code song song ngay. Chưa chạy hàng loạt khi giao thức chưa được duyệt.
>
> Các tên module, API, câu lệnh và mẫu log dưới đây là **đề xuất minh họa**, không phải code đã được xác minh trong repository. Ưu tiên tái sử dụng code hiện có. Không thay đổi dữ liệu đóng băng hoặc `configs/seq.yaml`.

## 1. Đọc phần này trước khi bắt đầu

| | Người 2 — Task 05 | Người 3 — Task 10 |
|---|---|---|
| Nhiệm vụ | Làm pipeline huấn luyện nền | Thêm hai cách cập nhật trọng số cho Seq |
| Phương pháp | DFT, C0, Seq | Seq+LLRD, Seq+LoRA |
| Việc làm ngay | Đọc code cũ, xây loader, trainer, lưu kết quả, test | Đọc kiến trúc ViSoBERT, viết module LLRD/LoRA, test độc lập |
| Cần đợi | Task 03–04 để khóa dữ liệu và giao thức | API Trainer và Task 04 để tích hợp; Task 07 để dò LLRD theo LR Seq |
| Bàn giao đầu tiên | Code nền và log kiểm thử | Module cải tiến và log kiểm thử |
| Khi chạy GPU | Task 06–09 (không phải Task 05) | Sau khi đủ điều kiện, theo Task 10 |

### Hiểu các phương pháp bằng một ví dụ

Chiều A: nguồn UIT-VSMEC, đích NEU-ESC. Giả sử đích chỉ có 10% train (khoảng 2.305 câu).

- **DFT:** ViSoBERT tiền huấn luyện → huấn luyện trên 2.305 câu NEU-ESC.
- **Seq:** checkpoint ViSoBERT đã học 100% UIT-VSMEC → thay đầu phân loại cho NEU-ESC theo giao thức → học 2.305 câu NEU-ESC.
- **C0:** ViSoBERT → học trên tập con NEU-ESC → tiếp tục học trên chính tập con đó, theo lịch đối chứng đã duyệt. Đây là phương án low-resource đề xuất; Task 04 phải xác nhận. Không dùng checkpoint đã thấy 100% NEU-ESC.
- **Seq+LLRD:** giống Seq về nguồn và dữ liệu, nhưng tầng thấp có tốc độ học nhỏ hơn tầng cao.
- **Seq+LoRA:** giống Seq về nguồn và dữ liệu, nhưng đóng băng trọng số encoder gốc; chỉ học adapter và đầu phân loại mới.

Chiều B làm ngược lại: NEU-ESC → UIT-VSMEC (khoảng 554 câu ở mức 10%, 7 nhãn đích). Tất cả phương pháp cùng chiều, mức và seed phải dùng cùng danh sách ID train đích, cùng dev/test và quy ước nhãn. Nguồn Seq luôn là checkpoint nguồn 100% cùng seed.

---

# PHẦN A — TASK 05: Chuẩn bị code DFT, C0, Seq

## A1. Mục tiêu và phạm vi

Task 05 là việc **chuẩn bị phần mềm**, không phải chạy 78 lượt thực nghiệm. Sau Task 05, nhóm phải có một pipeline nhận cấu hình, đọc đúng dữ liệu, khởi tạo đúng mô hình, train, chọn checkpoint theo dev và xuất kết quả có thể kiểm tra. Task 06 mới là smoke test GPU, Task 07 mới dò LR chính thức.

Không cần cài LLRD, LoRA, XLM-R hay LLM để hoàn thành Task 05.

## A2. Bước 1 — Đọc và kiểm kê code cũ

- [ ] Xác định entrypoint chạy DFT/REF, C0, Seq hiện tại.
- [ ] Tìm hàm đọc dữ liệu, khởi tạo encoder/head, optimizer, trainer, evaluation và resume.
- [ ] Xác định format checkpoint, cách lưu tokenizer, mapping nhãn, seed và hash.
- [ ] Đọc `AGENTS.md`, `seq_notes.md` nếu có; kiểm tra nhánh dự án thực tế.
- [ ] Ghi phần nào dùng lại, phần nào cần bổ sung, phần nào đang chưa rõ.

Workflow kỹ thuật cũ gợi ý `src/seq/trainer.py`, `src/seq/stage2.py`; kiểm tra chúng thực sự tồn tại trước khi sửa. Có thể tạo `src/phase2/` nếu phù hợp cấu trúc dự án. Không viết lại Trainer từ đầu nếu code cũ đã đáp ứng.

**Đầu ra:** bảng kiểm kê code ngắn (đường dẫn, chức năng, thay đổi cần thiết).

## A3. Bước 2 — Đọc dữ liệu theo mức và seed

Người điều phối bàn giao các file chỉ mục từ Task 03, dự kiến nằm trong `data/subsets_phase2/`, cùng MANIFEST và mapping nhãn.

Ví dụ: `neuesc_f10_s0_ids.csv` chứa ID train của NEU-ESC 10%, seed 0 (tên thật theo bàn giao).

Luồng loader:

```text
Đọc train gốc + danh sách ID được duyệt
    → xác minh hash và ID hợp lệ
    → lọc đúng ID (không lấy lại mẫu ngẫu nhiên)
    → lấy cột text chưa tách từ
    → tokenize bằng tokenizer ViSoBERT, max_length=256
    → tạo train loader; dev/test cố định riêng
```

- [ ] Báo lỗi nếu thiếu ID, ID trùng, sai hash hoặc ID/hash văn bản giao với dev/test.
- [ ] Kiểm tra đúng số câu và phân phối nhãn theo đầu ra Task 03; không tự thay đổi tập con.
- [ ] Giữ emoji; không đưa `text_seg` có dấu nối âm tiết `_` vào ViSoBERT. Đối chiếu nội dung đầu vào với cột raw đã duyệt, kiểm tra tỷ lệ `_`, ghi tỷ lệ cắt cụt.
- [ ] DFT/C0/Seq cùng `(đích, mức, seed)` trả về cùng tập ID/hash. C0 cả hai pha tuân thủ giới hạn dữ liệu đã duyệt.

Có thể viết test bằng dữ liệu giả nhỏ ngay khi chưa nhận file thật. Không dùng tập giả để chạy kết quả nghiên cứu.

## A4. Bước 3 — Khởi tạo mô hình đúng theo phương pháp

| Phương pháp | Encoder ban đầu | Head | Dữ liệu được học |
|---|---|---|---|
| DFT | `uitnlp/visobert` | Mới, số lớp đích | Train đích con |
| Seq | Checkpoint nguồn 100%, cùng seed | Head đích theo Task 04 (thường mới) | Nguồn 100% đã học trước + train đích con |
| C0 | Pha đầu từ mô hình tiền huấn luyện; pha sau từ checkpoint pha đầu | Theo giao thức C0 đã duyệt | Cùng tập đích con ở cả hai pha theo đề xuất |

Ví dụ khi Seq chiều A seed 2: nạp encoder từ checkpoint UIT-VSMEC seed 2, không nạp nhầm checkpoint NEU-ESC hoặc classifier 7 lớp vào bài toán 3 lớp. Với chiều B, head cần 7 đầu ra.

**C0 là điểm cần xác nhận đặc biệt:** lịch train, ngân sách cập nhật, việc giữ/reset head và khả năng tái sử dụng DFT làm pha đầu phải theo Task 04. Nếu tái sử dụng DFT, kiểm tra tập con, seed, cấu hình và checkpoint thực sự tương thích. Không tự tuyên bố C0 đã cân bằng số bước. Ghi provenance của cả hai pha. Nếu quyết định còn mở, chỉ chuẩn bị interface và test, không khóa hành vi chính thức.

## A5. Bước 4 — Chuẩn hóa Trainer dùng chung

Luồng một lượt chạy:

```text
Đọc config → set seed → loader → khởi tạo model
→ optimizer/scheduler → train → chấm dev mỗi epoch
→ early stopping → khôi phục best checkpoint
→ nếu là chạy chính: dự đoán test và lưu kết quả
→ nếu là sweep/smoke: chỉ xuất dev
```

Cấu hình tham chiếu kế hoạch: batch hiệu dụng 32 (ví dụ batch 16, tích lũy gradient 2), warmup 10%, weight decay 0,01, max 10 epoch, patience 2, max_length 256, bf16 nếu GPU hỗ trợ. LR và tiêu chí dev lấy từ cấu hình được duyệt; LR chính thức chỉ khóa sau Task 07.

Nếu Task 04 duyệt Macro-F1 5 lớp cho UIT-VSMEC, tính F1 từng lớp từ dự đoán 7 lớp trên toàn bộ dev, rồi lấy trung bình 5 lớp được chọn; không lọc bỏ mẫu Surprise/Other khi tính nhầm sang một bài toán 5 lớp khác. NEU-ESC dùng tiêu chí V3-NEU nếu được duyệt. Không dùng test để chọn checkpoint, LR hoặc sửa giao thức.

**Ví dụ interface chung (giả mã, không phải API có sẵn):**

```python
model = build_model(config)   # DFT / C0 / Seq
train, dev, test = load_splits(config)
optimizer = build_optimizer(model, config)
result = fit(model, optimizer, train, dev, config)
restore_best_checkpoint(model, result)
if config.run_mode == 'main':
    save_predictions(model, test, config)
save_run_metadata(result, config)
```

C0 cần orchestration hai pha riêng; đoạn trên chỉ minh họa một pha train. Nên để `build_optimizer` và `build_model` có điểm mở rộng rõ cho người phụ trách Task 10, không buộc người đó sao chép Trainer.

## A6. Bước 5 — Lưu kết quả và hỗ trợ chạy lại

Mỗi run có ID duy nhất, ví dụ `SA_seq_f10_s0` (thêm LR hoặc hash cấu hình cho sweep). Đường dẫn dưới đây là gợi ý, thống nhất với nhóm trước khi dùng.

```text
results/phase2/<run_id>/
  config.yaml          # cấu hình đã dùng
  metadata.json        # seed, data hash, checkpoint nguồn, model, phiên bản
  metrics.json         # dev; test chỉ khi chạy chính
  predictions.csv      # ID test, nhãn thật, nhãn dự đoán, p từng lớp
  cost.json            # thời gian, GPU, VRAM, tham số
  completion.json      # trạng thái và hash đầu ra hoàn chỉnh
```

Lưu best checkpoint theo quy ước repo. Dự đoán V7 của UIT-VSMEC phải giữ xác suất 7 lớp để tính lại các góc nhìn nhãn; dùng mapping dự án để xuất V3 nếu cần. Xác suất hữu hạn, không âm, tổng gần 1 trong sai số số học; ID duy nhất và khớp test.

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Ghi `train_seconds`, `gpu_hours`, `peak_vram_mb`, `gpu_name`; thống nhất phạm vi đo, số GPU và đơn vị. Với một GPU, GPU-giờ = giây sử dụng GPU / 3600. C0 cần ghi riêng chi phí từng pha và tổng/chi phí tăng thêm nếu dùng lại checkpoint.

- Resume phải khôi phục đủ trạng thái mô hình, optimizer, scheduler, tiến độ và RNG/sampler cần thiết; nếu chưa hỗ trợ giữa epoch, ghi rõ hạn chế và không tuyên bố test đã đạt.
- SKIP chỉ khi run đã hoàn chỉnh và hash/config khớp; thiếu prediction của run chính thì không được bỏ qua. Sweep không yêu cầu file test.

## A7. Bước 6 — Unit test và nghiệm thu

| Test | Cách kiểm tra | Kết quả mong đợi |
|---|---|---|
| Loader | ID giả có trùng/thiếu/sai hash | Báo lỗi; ID hợp lệ lọc đúng |
| Lồng nhau | So sánh tập ID các mức từ Task 03 | 10% nằm trong 25%, 25% nằm trong 50% |
| Chống rò rỉ | Cố tình đưa ID/hash dev/test vào train | Từ chối chạy |
| Khởi tạo | Hai đích 3/7 lớp, seed/checkpoint khác nhau | Đúng encoder, head và provenance |
| C0 | Đưa checkpoint đã thấy toàn bộ đích vào C0 10% | Từ chối theo giao thức low-resource nếu được duyệt |
| Dev | Tạo các epoch có metric dev giả khác nhau | Khôi phục đúng best checkpoint |
| Sweep | Chạy mode sweep trên mock | Không gọi hàm đánh giá test |
| Resume | Ngắt và tiếp tục trên mô hình nhỏ | Trạng thái phục hồi đúng theo phạm vi hỗ trợ |
| Tham số | Model có một phần đóng băng | Tổng và trainable đếm đúng |
| SKIP | Xóa file đầu ra bắt buộc | Không bỏ qua run thiếu |

Chạy `python -m pytest -q` bằng môi trường dự án (Windows có thể dùng `.venv/Scripts/python.exe -X utf8 -m pytest -q` như workflow cũ). Đây là câu lệnh đề xuất, cần kiểm tra cấu trúc môi trường thật.

### Checklist bàn giao Task 05

- [ ] PR/commit code DFT/C0/Seq; hướng dẫn cách gọi thực tế.
- [ ] Loader nhận dữ liệu Task 03; cấu hình Task 04 không bị tự ý thay đổi.
- [ ] Log unit test cũ và mới đều PASS, hoặc ghi blocker chưa đạt (không đánh dấu hoàn thành).
- [ ] Mô tả interface tích hợp LLRD/LoRA và format kết quả cho người 3.
- [ ] Ví dụ một config chạy thử DFT NEU-ESC 10%, seed 0, chỉ dev, sẵn sàng Task 06.

**Task 05 kết thúc tại đây.** Không cần chứng minh mô hình đạt F1 cao, không chạy 78 cấu hình, không cần chờ Task 10.

---

# PHẦN B — TASK 10: Thực nghiệm Seq+LLRD và Seq+LoRA

## B1. Mục tiêu

Giai đoạn trước, Seq−DFT của ViSoBERT ở 100% là −0,0441 chiều A và +0,0635 chiều B. Nhóm muốn biết liệu thay đổi cách cập nhật encoder ở pha thích nghi đích có giảm chuyển giao âm hoặc giảm dao động giữa seed hay không.

Không thay mô hình nguồn, không tạo thêm dữ liệu, không áp dụng LLRD/LoRA cho DFT trong phạm vi hiện tại. Hai phương pháp là hai nhánh riêng, không kết hợp LLRD+LoRA.

## B2. Bước 1 — Hiểu và viết LLRD

**LLRD (Layer-wise Learning Rate Decay)** vẫn train toàn bộ encoder và head, nhưng tốc độ học của tầng thấp nhỏ hơn. Ý tưởng là hạn chế thay đổi mạnh các biểu diễn đã học; đó là giả thuyết cơ chế, chưa phải bằng chứng giảm forgetting.

Theo workflow kỹ thuật: head nhận LR đỉnh η; encoder tầng l (đếm từ 0 ở dưới lên, có L tầng) nhận `η * ξ ** (L-l)`; embeddings nhận `η * ξ ** (L+1)`.

Ví dụ minh họa với L=12, LR đỉnh 2e-5, ξ=0,8 (không phải LR đã chọn):

| Khối | LR xấp xỉ |
|---|---:|
| Head | 2,00e-5 |
| Tầng 11 (trên cùng) | 1,60e-5 |
| Tầng 6 | 5,24e-6 |
| Tầng 0 (dưới cùng) | 1,37e-6 |
| Embeddings | 1,10e-6 |

**Cần lập trình:** tạo các nhóm tham số cho optimizer; mỗi nhóm có LR theo công thức. Xác định số tầng và tên module thật từ model, không giả định hard-code 12 cho mọi kiến trúc. Tách bias/LayerNorm không weight decay theo quy tắc dự án. Kiểm tra các tham số khác (nếu có) được phân nhóm tường minh.

Giả mã:

```python
# l là chỉ số tầng encoder, L lấy từ model thật
lr_layer = peak_lr * decay ** (L - l)
# tạo optimizer groups cho head, từng tầng, embeddings
# mỗi tham số trainable phải xuất hiện đúng một lần
```

- [ ] Test mọi tham số trainable nằm đúng một group, không trùng/thiếu.
- [ ] Test LR embeddings < tầng thấp < tầng cao < head; kiểm tra số cụ thể với ξ=0,8 và 0,9.
- [ ] Test weight decay và scheduler giữ tỷ lệ LR giữa các nhóm theo giao thức.
- [ ] Test optimizer cập nhật encoder/head trên mô hình nhỏ; không vô tình đóng băng encoder.

## B3. Bước 2 — Hiểu và viết LoRA

**LoRA (Low-Rank Adaptation)** giữ nguyên trọng số gốc của encoder đã học từ miền nguồn, thêm hai ma trận nhỏ A và B vào các phép biến đổi tuyến tính được chọn. Minh họa: trọng số hiệu dụng `W = W0 + (alpha/r) * B @ A`. W0 đóng băng; A/B trainable. Head phân loại đích vẫn được train đầy đủ.

| Tham số | Giá trị kế hoạch |
|---|---|
| Module đích | query, value |
| Rank r | 8 hoặc 16 |
| Alpha | 2r (16 hoặc 32) |
| Dropout | 0,1 |
| LR dò | 1e-4, 3e-4, 5e-4 |
| Epoch | Tối đa 10; có thể nâng 15 nếu chưa hội tụ trên dev theo quy tắc duyệt |

**Cần lập trình:** kiểm tra tên module query/value và head thật của ViSoBERT; có thể dùng PEFT nếu tương thích môi trường dự án. Nạp checkpoint nguồn và tạo head đích trước khi gắn adapter; đóng băng encoder gốc và kiểm tra head train đầy đủ. Không thêm adapter vào nhầm module hoặc giữ nhầm head nguồn.

Ví dụ cấu hình PEFT minh họa (chưa kiểm tra version/API thực tế):

```python
from peft import LoraConfig, TaskType, get_peft_model
cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=['query', 'value'],
    modules_to_save=['classifier'],  # chỉ dùng nếu đúng tên head thật
    bias='none',
)
model = get_peft_model(model, cfg)
```

Không dùng đoạn code trên nguyên xi nếu cấu trúc head/module khác. Khóa phiên bản dependency theo repo. Kiểm tra lưu/nạp adapter + head, liên kết đúng base checkpoint nguồn cùng seed; lưu metadata để tái lập.

- [ ] Liệt kê tên và số lượng module được gắn adapter; không chấp nhận số module bằng 0.
- [ ] Liệt kê tham số trainable: adapter và toàn bộ head đích; kiểm tra không có tham số encoder gốc trainable ngoài phạm vi.
- [ ] Sau vài bước train trên dữ liệu giả, so sánh tensor/hash trọng số encoder gốc trước/sau: không đổi. Kiểm tra adapter/head có cập nhật (không yêu cầu mọi tensor thay đổi ở bước đầu).
- [ ] Kiểm tra `0 < trainable_params < total_params` và đếm chính xác.
- [ ] Lưu/nạp adapter và head; ở chế độ eval, dự đoán khớp trước/sau trong sai số số học.

## B4. Bước 3 — Tích hợp với Task 05

Hai người thống nhất interface trước khi sửa cùng file. Người 2 quản lý pipeline nền; người 3 quản lý module LLRD và LoRA. Ví dụ bố cục (đề xuất, không bắt buộc):

```text
src/phase2/
  trainer.py        # người 2: pipeline chung
  data.py           # người 2: đọc ID
  llrd.py           # người 3: tạo optimizer groups
  lora.py           # người 3: gắn adapter, save/load
```

- LLRD thay cách xây optimizer của Seq, không thay loader hay evaluator.
- LoRA thay cách chuẩn bị model trainable của Seq và cách lưu/khôi phục checkpoint; dùng pipeline chung.
- Mọi phương pháp đọc cùng config đã duyệt, hash tập con và nguồn checkpoint, cùng tiêu chí dev, format prediction và cost log.
- Test riêng bằng model nhỏ trước, sau đó test tích hợp. Chỉ dùng GPU smoke khi có pipeline và giao thức ổn định; ghi chi phí ngoài ngân sách chính.

**Đây là toàn bộ phần người 3 có thể chuẩn bị ngay.** Chưa cần đợi kết quả Task 08–09 để viết code. Không chạy sweep LLRD trước khi Task 07 khóa LR Seq.

## B5. Bước 4 — Dò siêu tham số (sau Task 07)

Chỉ dùng train 10% + dev, seed 0, hai chiều. LR Seq của mỗi chiều lấy từ `selection_p1_lr.json` đã khóa.

| Phương pháp | Cấu hình mỗi chiều | Hai chiều |
|---|---|---:|
| LLRD | ξ=0,8; ξ=0,9, dùng LR đỉnh Seq | 4 lượt |
| LoRA | r=8/16 × LR=1e-4/3e-4/5e-4 | 12 lượt |
| Tổng | 8 cấu hình mỗi chiều | 16 lượt |

Chọn cấu hình theo metric dev được duyệt cho từng chiều/phương pháp. Chốt trước quy tắc xử lý hòa, nghiệm ở biên và LoRA chưa hội tụ; ghi thêm lượt dò nếu mở rộng. Không xem test trong sweep. Lưu `results/phase2/selection_p2_hparams.json` cùng bảng điểm dev, config/hash và lý do lựa chọn. Không chọn phương pháp dựa trên điểm test rồi thay đổi phạm vi đã đăng ký.

## B6. Bước 5 — Chạy thực nghiệm chính

| Đích | Mức | Phương pháp | Seed | Lượt |
|---|---:|---|---|---:|
| NEU-ESC (A) | 10% | LLRD, LoRA | 0–4 | 10 |
| UIT-VSMEC (B) | 10% | LLRD, LoRA | 0–4 | 10 |
| NEU-ESC (A) | 100% | LLRD, LoRA | 0–4 | 10 |
| Tổng chính | | | | 30 |

Dùng siêu tham số khóa từ 10%; mỗi lượt khởi tạo từ checkpoint nguồn đúng seed. Dự toán cơ sở Task 10: 16 dò + 30 chính = 46 lượt; chưa tính smoke, mở rộng lưới, chạy lại mốc cũ nếu không tương thích. 25% là tùy chọn (+20 lượt nếu cả hai phương pháp/hai chiều), chỉ chạy khi nhóm duyệt.

Có thể chạy song song với Task 08–09 khi Task 04, 05, 07 và test tích hợp đã đạt. Không cần đợi Task 09, nhưng cần Seq cùng mức/seed/giao thức để phân tích hiệu quả.

## B7. Bước 6 — Đọc kết quả và bàn giao

Với mỗi chiều/mức, tính Macro-F1 trung bình và SD trên 5 seed, ghép cặp cùng seed và ID test. So sánh:

- `Seq+LLRD − Seq` và `Seq+LoRA − Seq`: mức cải thiện so với Seq thường.
- `Seq+LLRD − DFT` và `Seq+LoRA − DFT`: còn chuyển giao âm không?
- SD mới so với SD Seq: mức dao động giữa seed. CI bootstrap cho tỷ số SD thực hiện theo Task 12; nếu mẫu số bằng 0 thì ghi không xác định.

Ví dụ **giả lập, không phải kết quả nghiên cứu**: DFT=0,70; Seq=0,65; LLRD=0,68. Seq âm −0,05 so với DFT; LLRD tăng +0,03 so với Seq nhưng vẫn âm −0,02 so với DFT. Như vậy chỉ có thể nói giảm độ lớn chuyển giao âm ở điểm ước lượng, chưa loại bỏ chuyển giao âm. Cần CI và phân tích seed để kết luận thận trọng.

Không so sánh trực tiếp với mốc Seq 100% cũ nếu giao thức không tương thích (Task 04 quyết định). Với 5 seed, kết luận về SD chỉ mang tính thăm dò. Không khẳng định giảm forgetting nếu chưa đo trực tiếp; thiếu đối chứng DFT+LLRD/LoRA nên chưa tách được lợi ích điều chuẩn thuần túy khỏi lợi ích bảo toàn tri thức nguồn.

### Checklist bàn giao Task 10

- [ ] Code LLRD/LoRA, cấu hình và log unit test/tích hợp.
- [ ] Bằng chứng encoder gốc LoRA không thay đổi, head và adapter được học; LLRD có LR đúng từng tầng.
- [ ] Bảng 16 lượt dò và file khóa siêu tham số (ghi lượt bổ sung nếu có).
- [ ] 30 lượt chạy chính hợp lệ theo phạm vi cơ sở hoặc quyết định giảm phạm vi được duyệt.
- [ ] Dự đoán test theo ID, log chi phí, bảng F1/Gain/SD theo chiều, mức và seed.
- [ ] Ghi kết quả âm/dương, hạn chế và lỗi còn tồn tại; bàn giao cho Task 12.

---

## 2. Quy tắc phối hợp và bàn giao giữa ba người

| Mốc | Người điều phối | Người 2 | Người 3 |
|---|---|---|---|
| Hiện tại | Hoàn thành 03–04 | A2–A3, chuẩn bị Trainer/test | B2–B3, test module độc lập |
| Khi Task 04 được duyệt | Bàn giao config, mapping, ID/hash và quyết định C0/dev | Hoàn thiện A4–A7 | Tích hợp B4 theo API chung |
| Khi Task 05 đạt | Duyệt đầu ra | Task 06 smoke, Task 07 dò LR | Kiểm thử tích hợp, chuẩn bị sweep |
| Khi LR đã khóa | Kiểm tra config | Task 08–09 | B5–B6 (Task 10), song song nếu có GPU |
| Khi có kết quả | Tổng hợp Task 12 | Bàn giao DFT/C0/Seq | B7 và bàn giao kết quả cải tiến; Task 11 theo phân công riêng |

Đề xuất mỗi người làm nhánh riêng, mở PR nhỏ, không sửa cùng một Trainer cùng lúc. Người điều phối duyệt thay đổi giao thức; không tự đổi seed, dữ liệu, head, tiêu chí dev hoặc siêu tham số đã khóa. Mọi thay đổi có ảnh hưởng khoa học phải ghi vào `reports/phase2/decisions.md` hoặc nơi nhóm thống nhất.

### Mẫu cập nhật tiến độ ngắn

```text
Task / người phụ trách:
Đã làm:
Commit hoặc đường dẫn đầu ra:
Test đã chạy / kết quả:
Vướng mắc:
Cần ai bàn giao gì:
Việc tiếp theo:
```

## 3. Việc giao ngay hôm nay

**Người 2:** mở repository, tìm entrypoint DFT/REF/C0/Seq, lập bảng kiểm kê code và chuẩn bị loader nhận file ID; viết test giả lập. Sau đó thống nhất API Trainer với người 3.

**Người 3:** mở cấu trúc ViSoBERT và code Seq, xác định embeddings, các tầng encoder, query/value và head; viết module LLRD/LoRA cùng unit test độc lập. Chưa chạy sweep.

**Người điều phối:** hoàn thành Task 03–04, cung cấp file ID/hash, cấu hình và các quyết định giao thức. Không yêu cầu hai người chờ toàn bộ nghiên cứu mới bắt đầu code.

**Giới hạn tài liệu:** Chưa kiểm tra code, dữ liệu, checkpoint, log Task 01–02 hay môi trường thực tế. Hướng dẫn không xác nhận các module hoặc API ví dụ tồn tại, không xác nhận test đã PASS, và không thay thế các quyết định khoa học tại Task 04.
