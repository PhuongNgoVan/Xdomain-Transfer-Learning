# Biên bản Quyết định Giao thức Thực nghiệm Giai đoạn 2 (`phase2-v1` — Task 04)

- **Phiên bản giao thức:** `phase2-v1` (`protocol_tag: p2v1`)
- **File cấu hình thực thi:** [configs/phase2.yaml](../../configs/phase2.yaml)
- **Module kiểm tra hợp đồng & thống kê:** [src/phase2/protocol.py](../../src/phase2/protocol.py)
- **Trạng thái:** **ĐÃ KHÓA CHO TRIỂN KHAI TASK 05–12** (không dùng điểm `test` để lựa chọn bất kỳ mục nào; `configs/seq.yaml`, `data/clean/`, `data/splits/` giữ nguyên bất biến).

---

## 1. Bảng tóm tắt 6 quyết định đã chốt

| # | Hạng mục | Quyết định khóa chính thức (`phase2-v1`) |
|---|---|---|
| **1** | **`C0 low-resource`** | Tái sử dụng best checkpoint `DFT` cùng `(đích, mức f%, seed s ∈ {0,1,2})` làm Pha 1; Pha 2 nạp encoder, **khởi tạo lại head mới có trọng số ban đầu trùng `Seq` cùng seed**, **tạo mới hoàn toàn `optimizer` và `scheduler`**, train tiếp trên chính tập con `f%` với LR & lịch train của `Seq` giai đoạn 2. |
| **2** | **`Dev` & chọn checkpoint** | Giữ nguyên `dev` gốc (`VSMEC: 686 câu`; `NEU-ESC: 3.305 câu`) và `test` gốc. `evals_per_epoch = 1` (cuối mỗi epoch), `patience = 2`, `warmup = 10%`, `epochs = 10` (`LoRA` nâng lên `15` theo quy tắc khách quan trên `dev seed 0`). Chọn best bằng **`Macro-F1 5 lớp`** cho VSMEC và **`Macro-F1 V3-NEU`** cho NEU-ESC. |
| **3** | **Mốc `100%`** | Giữ số liệu Giai đoạn 1 (`ViSoBERT`, `PhoBERT`) làm **tham chiếu lịch sử** (vẽ nét đứt + ghi chú khác biệt giao thức). **Bắt buộc chạy bổ sung `5 DFT + 5 Seq` ở `100% chiều A` (`seed 0..4`) theo `phase2-v1`** (+10 lượt) trước hoặc cùng Task 10. |
| **4** | **Seed & Thống kê** | `DFT, Seq, LLRD, LoRA`: `seed 0..4`; `C0, XLM-R`: `seed 0..2` (`Seq − C0` ghép 3 seed chung). Paired bootstrap hai tầng (`B=10.000, boot_seed=0, share_test=True, share_seeds=True`), **Holm `m = 2` ($\alpha = 0,05$)** cho `Seq − DFT` ở `10%` chiều A (`V3-NEU`) và chiều B (`V7-VSMEC`). |
| **5** | **Phạm vi & Ngân sách** | Gói cơ sở mới: **`164` lượt train** (`ViSoBERT` 10/25/50% + `10` lượt 100% chiều A + `46` lượt LLRD/LoRA + `18` lượt chính XLM-R). Các mục `5%` (+26), Pha 2 ở `25%` (+20), `mDeBERTa-v3` (+18) để ở chế độ **tùy chọn tạm hoãn (`OPTIONAL_DEFERRED`)**. |
| **6** | **Tên gọi & `run_id`** | Dùng thống nhất **`DFT`** trong báo cáo (ánh xạ rõ `REFN2N / REFV2V` cũ). Mọi `run_id` GĐ2 gắn nhãn phiên bản `p2v1` và mã băm 8 ký tự `h{cfg_hash8}` để chống tái sử dụng nhầm kết quả cũ. |

---

## 2. Đặc tả kỹ thuật chi tiết từng quyết định

### Quyết định 1 — Đối chứng `C0 low-resource`
- **Pha 1:** Tại mức $f\% \in \{10, 25, 50\}$ và $s \in \{0, 1, 2\}$, tái sử dụng trực tiếp best checkpoint của lượt chạy `DFT` tương ứng ở chính `(đích, f%, s)`. Tuyệt đối nghiêm cấm nạp checkpoint đã huấn luyện trên `100%` miền đích cho bất kỳ thực nghiệm `f% < 100%` nào (`forbid_100pct_target_checkpoint_below_100: true`).
- **Pha 2:**
  1. Kiểm tra khớp `ids_sha256`, `source_files_sha256` và cấu hình của checkpoint `DFT` Pha 1 trước khi nạp.
  2. Nạp trọng số `encoder` từ checkpoint `DFT` Pha 1; **khởi tạo mới `classifier head`** bằng `set_seed(seed)` trên cùng kích thước `hidden_size` và `num_labels` để trọng số ban đầu của `head` trùng khớp từng bit với `Seq` giai đoạn 2 cùng `seed`.
  3. **Khởi tạo mới hoàn toàn `AdamW` optimizer và linear warmup scheduler** (không nạp trạng thái momentum/step từ Pha 1).
  4. Huấn luyện trên chính tập con `f%` cùng `seed s` với cùng LR đã khóa của `Seq` giai đoạn 2 ở chiều tương ứng (`lr_mode: reuse_seq_selected`).
- **Ghi log & giới hạn diễn giải:** Lưu đầy đủ `stage1_best_step` (từ `DFT`) và `stage2_best_step` / `total_update_steps` thực tế. Khi viết báo cáo, nêu rõ việc tái sử dụng `DFT` tránh được `18` lượt train Pha 1 dư thừa (giữ lõi Pha 1 ở `78` lượt), đồng thời lưu ý `C0` không cô lập hoàn toàn yếu tố "tổng số bước cập nhật lịch sử" vì Pha 1 của `Seq` học trên `100%` nguồn nhiều bước hơn Pha 1 của `C0` học trên `f%` đích.

### Quyết định 2 — Tập `dev`, Early Stopping và Tiêu chí chọn Checkpoint
- **Dữ liệu đánh giá:** Giữ cố định 100% `dev` gốc (`UIT-VSMEC: 686 câu ≥ 172`; `NEU-ESC: 3.305 câu`) và `test` gốc (`692` / `6.613` câu) cho mọi mức tài nguyên. Chỉ giảm tập `train` theo đúng 40 file chỉ mục lồng nhau đã khóa ở Task 03 (`data/subsets_phase2/`).
- **Lịch huấn luyện & dừng sớm:**
  - `batch_size = 16`, `gradient_accumulation_steps = 2` (batch hiệu dụng `32`), `warmup_ratio = 0.10`, `weight_decay = 0.01`, `max_grad_norm = 1.0`, `dropout = 0.1`, `max_length = 256`, `precision = bf16` (`strict_precision = true`).
  - Đánh giá **1 lần vào cuối mỗi epoch** (`evals_per_epoch = 1`), `patience = 2` epoch, tối đa `10` epoch. *(Ghi chú phương pháp: ở mức dữ liệu thấp như 10% VSMEC chỉ có ~18 step/epoch, việc đánh giá 2 lần/epoch kèm `patience = 2` có thể gây dừng sớm do dao động nhiễu ngắn hạn, dù không phải chắc chắn dừng sau 1 epoch; đánh giá cuối mỗi epoch giúp ổn định chọn mô hình).*
  - **Quy tắc khách quan nâng trần epoch cho `LoRA` (`unconverged_extension`):** Trong lượt dò `LoRA` ở mức `10%` trên `seed 0`, nếu cấu hình tối ưu trên `dev` đạt `best_epoch ∈ {9, 10}` và chưa kích hoạt early stopping khi hết epoch 10 thì nâng `max_epochs = 15` cho nhánh `LoRA` và ghi nhận tường minh vào metadata + báo cáo; ngược lại giữ `10` epoch.
- **Tiêu chí chọn best checkpoint trên `dev` (áp dụng đồng nhất cho mọi phương pháp):**
  - **Đích `UIT-VSMEC`:** `macro_f1_v5` — mô hình vẫn dự đoán đầy đủ **7 lớp** trên toàn bộ `686` câu `dev` (không lọc bỏ mẫu `Surprise`/`Other` khỏi tập `dev` để các dự đoán nhầm vào/ra hai lớp này vẫn phản ánh đúng vào Precision/Recall của 5 lớp kia), sau đó lấy trung bình cộng F1 của 5 lớp `{Enjoyment(0), Sadness(1), Fear(2), Anger(3), Disgust(4)}`.
  - **Đích `NEU-ESC`:** `macro_f1_v3` (`V3-NEU`, trung bình F1 của 3 lớp `{negative(0), neutral(1), positive(2)}` trên toàn bộ `3.305` câu `dev`).
  - **Thước đo báo cáo chính trên `test`:** Giữ nguyên `V7-VSMEC` (cho chiều B) và `V3-NEU` (cho chiều A); các góc nhìn khác báo cáo song song ở Task 12.

### Quyết định 3 — Xử lý Mốc `100%`
- **Mốc Giai đoạn 1:** Giữ toàn bộ kết quả `100%` của `ViSoBERT` và `PhoBERT` từ Giai đoạn 1 làm **tham chiếu lịch sử**, biểu diễn bằng đường nét đứt / ký hiệu riêng trên biểu đồ và ghi rõ các điểm khác biệt giao thức xác nhận từ log (`warmup 6% vs 10%`, `patience 5/eval×2 vs patience 2/eval×1`, `dev V7 vs V5`).
- **10 lượt chạy bổ sung bắt buộc ở `100%` chiều A (`SA: UIT-VSMEC → NEU-ESC`):**
  - Chạy mới **`5 lượt DFT` (`seed 0..4`)** và **`5 lượt Seq` (`seed 0..4`)** ở `100%` NEU-ESC theo đúng giao thức `phase2-v1` (dùng lại LR đã khóa từ `dev` mức 10% ở Task 07), thực hiện trước hoặc song song với Task 10.
  - Mục đích: Đảm bảo tại `100%` chiều A có đủ bộ đối chứng đồng nhất giao thức (`DFT`, `Seq`, `Seq+LLRD`, `Seq+LoRA`) để đánh giá chính xác cả cải thiện so với `Seq` lẫn mức độ thu hẹp *negative transfer* (`Seq* − DFT`).
- **Giới hạn ghi nhận rõ trong báo cáo:** `C0 100%` chiều A (3 lượt) và cụm `DFT/Seq/C0 100%` chiều B (`5 + 5 + 3 = 13` lượt) chưa chạy lại theo `phase2-v1` để tiết kiệm `16` lượt GPU (nếu chạy toàn bộ sẽ lên `180` lượt cơ sở); do đó không khẳng định "điểm đảo chiều đồng nhất giao thức tại 100%" cho các nhánh chưa chạy lại này.

### Quyết định 4 — Seed, Kiểm định thống kê & Kết quả xác minh Code trước khi khóa
- **Phân bổ seed:** `DFT, Seq, Seq+LLRD, Seq+LoRA` dùng `seed 0..4` (`5` seed); `C0` và `XLM-R` dùng `seed 0..2` (`3` seed); phép hiệu `Seq − C0` chỉ ghép trên đúng 3 seed chung `[0, 1, 2]`.
- **Gia đình xác nhận chính (`Holm m = 2, α = 0,05`):**
  1. `P2_SA_10`: `Seq − DFT` của `ViSoBERT` tại `10%` chiều A (`V3-NEU`, `macro_f1`).
  2. `P2_SB_10`: `Seq − DFT` của `ViSoBERT` tại `10%` chiều B (`V7-VSMEC`, `macro_f1`).
  - Nếu thiếu bất kỳ so sánh nào hoặc chưa đủ 5 seed ghép cặp thì để trống `p_holm` và `significant` (tuyệt đối không thu nhỏ $m$ xuống 1). Toàn bộ các so sánh khác là **thăm dò (`exploratory = True`)**.
- **Kết quả xác minh trực tiếp mã nguồn thống kê trước khi khóa (`tests/test_phase2_task04.py`):**
  1. `paired_gain` (`src/seq/analysis.py`) & `bootstrap_contrast` (`src/stats/bootstrap.py`): đã kiểm chứng bắt buộc trùng khít thứ tự `id` câu test và `seeds`, đồng bộ trọng số câu `Multinomial` (`share_test=True`) và chỉ số seed (`share_seeds=True`).
  2. `bootstrap_pvalue` (`src/stats/holm.py`): là p-value hai phía suy ra từ phân phối bootstrap ghép cặp (tail-area CI inversion) có hiệu chỉnh hữu hạn mẫu $(k+1)/(B+1)$, đảm bảo $p_{\text{boot}} \ge 2/(B+1) \approx 0,0002 > 0$ và dưới phân phối đối xứng quanh `0` cho $p_{\text{boot}} \approx 1,0$.
  3. `apply_phase2_holm` (`src/phase2/protocol.py`): tái sử dụng `holm_adjust` cho đúng gia đình $m = 2$ mà không sửa `src/seq/analysis.py` ($m = 4$ của `seq-v1`).
  4. `bootstrap_seed_sd_ratio` (`src/phase2/protocol.py`): xử lý tường minh trường hợp $\text{SD}_{\text{Seq}} < 10^{-8}$ trên 5 seed thật (trả `defined=False, point=NaN`, ghi `"không xác định"`) và đếm số lượt lặp bootstrap suy biến rút trùng 1 seed 5 lần (`~0,16%` khi $S=5$) vào `n_nan`.

### Quyết định 5 — Phạm vi và Ngân sách lượt huấn luyện (`164` lượt cơ sở)

| Nhóm công việc | Chi tiết cấu hình | Số lượt cơ sở | Ghi chú |
|---|---|---:|---|
| **Task 07:** Dò LR mức 10% | `ViSoBERT`: `DFT` (2 đích × 3 LR) + `Seq` (2 chiều × 3 LR), `seed 0`, dev-only | **12** | Khóa vào `selection_p1_lr.json` |
| **Task 08:** Thực nghiệm lõi 10% | `2 chiều × (5 DFT + 5 Seq + 3 C0)` | **26** | Hoàn tất 2 so sánh chính Holm $m=2$ |
| **Task 09:** Mở rộng lõi 25%, 50% | `2 chiều × 2 mức × (5 DFT + 5 Seq + 3 C0)` | **52** | Dùng lại LR từ Task 07 |
| **Mốc 100% chiều A (`phase2-v1`)** | `SA (VSMEC → NEU-ESC)` × `(5 DFT + 5 Seq)`, `seed 0..4` | **10** | Đối chứng cùng giao thức cho Task 10 |
| **Task 10:** `Seq+LLRD` & `Seq+LoRA` | `16` dò (10%, `seed 0`) + `20` chính (10%, 2 chiều) + `10` chính (100%, chiều A) | **46** | LLRD dùng LR đỉnh của Seq |
| **Task 11:** Tham chiếu `XLM-R-base` | `3 seed × (DFT 100% 2 miền + DFT 10% 2 miền + Seq 10% 2 chiều)` | **18** | Cộng `+12` lượt dev-only nếu dò LR riêng cho XLM-R |
| **Tổng gói cơ sở** | *(chưa gồm 1–2 lượt smoke Task 06 & suy luận ZS / LLM không train)* | **164** | |

- **Hạng mục tạm hoãn (`OPTIONAL_DEFERRED`):** Mức `5%` (`+26` lượt — đã đạt ngưỡng kỹ thuật `Surprise = 12 ≥ 5` ở Task 03 nhưng chỉ bật nếu còn GPU sau gói cơ sở), Pha 2 ở `25%` (`+20` lượt), `mDeBERTa-v3-base` (`+18` lượt), đồng nhất nốt 100% còn lại (`+16` lượt).
- **Thứ tự cắt giảm nếu thiếu GPU:** `5%` → `Pha 2 ở 25%` → `mDeBERTa-v3` → `mức 50%` → `giảm seed C0` (bảo vệ tuyệt đối 5 seed `DFT/Seq` ở `10%`).

### Quyết định 6 — Quy ước tên gọi và `run_id`
- Trong báo cáo và bảng tổng hợp dùng tên **`DFT`**; khi đọc file lịch sử Giai đoạn 1 ánh xạ tường minh `REFN2N → DFT_NEU_100_phase1` và `REFV2V → DFT_VSMEC_100_phase1`.
- Khuôn dạng `run_id` bắt buộc cho mọi lượt chạy `phase2-v1`:
  ```text
  {backbone}_{direction}_{method_tag}_f{fraction}_s{seed}_lr{lr}_{protocol_tag}_h{cfg_hash8}
  ```
  Ví dụ: `ViSoBERT_SA_dft_f10_s0_lr3e-5_p2v1_h8c41d09a` hoặc `ViSoBERT_SA_seq-lora-r8_f10_s0_lr0.0003_p2v1_h2f90ab11`.  
  *(Trong đó `method_tag` chỉ dùng chữ/số/dấu gạch ngang `-`, không chứa dấu gạch dưới `_`, và `cfg_hash8` là 8 ký tự đầu SHA-256 của cấu hình hiệu dụng).*
