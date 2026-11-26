# Báo cáo Task 01: Kiểm kê tài nguyên cũ (Phase 2)

> Sinh tự động (chế độ chỉ đọc, offline CPU) từ `scripts/run_phase2_task01_02.py`. Không chỉnh sửa dữ liệu, checkpoint hay cấu hình gốc.

## 1. Bảng kiểm kê tổng hợp

| Tài nguyên | Đường dẫn thực tế | Trạng thái | Ghi chú |
|---|---|---|---|
| Nhánh & nền Git | `phase2-resource-transfer` @ `29564eb` (gốc `seq-core` `62df4cd`) | **PRESENT** | Đúng nhánh 'phase2-resource-transfer' @ 29564eb, kế thừa trực tiếp từ seq-core (62df4cd). |
| UIT-VSMEC train/dev/test | `data/clean/vsmec_{train,dev,test}.csv` | **PRESENT** | train=5542, dev=686, test=692; đủ cột `id,text,label,subset,text_seg`. |
| NEU-ESC train/dev/test | `data/clean/neuesc_100.csv` (+ `data/raw/NEU_ESC_full.csv`) | **PRESENT** | train=23048, dev=3305, test=6613 (tổng=32966); raw có `NEU_ESC_full.csv` giữ nhãn gốc 4 lớp. |
| Tập chia nhỏ cũ (`data/splits/`) | `data/splits/neuesc_*.csv` (12 file) | **PARTIAL** | Chỉ có tập cắt NEU-ESC cũ (1%, 5%, 10% độc lập, không lồng nhau) — chưa có tập con lồng nhau 5/10/25/50% cho UIT-VSMEC và NEU-ESC (làm ở Task 03). |
| Checkpoint nguồn UIT-VSMEC (ViSoBERT, s0–s4) | `null (chưa chép về local)` | **MISSING** | Hợp lệ `0/5` seed (cần tải từ Drive trước Task 07/08). |
| Checkpoint nguồn NEU-ESC (ViSoBERT, s0–s4) | `null (chưa chép về local)` | **MISSING** | Hợp lệ `0/5` seed (cần tải từ Drive trước Task 07/08). |
| Code DFT / C0 / Seq / ZS / Stats | `src/seq/`, `src/zs/`, `src/stats/`, `src/plm/model_ckpt.py` | **PARTIAL** | Đủ bộ lõi `seq-v1` & `zs`; cần bổ sung DFT từ HF, C0 low-resource, LLRD/LoRA, bộ đếm chi phí & view mới (xem Mục 3). |
| Cấu hình giai đoạn 1 | `configs/base.yaml`, `step2_data.yaml`, `zs.yaml`, `seq.yaml` | **PRESENT** | Bảo toàn nguyên trạng; GĐ2 dùng file cấu hình riêng ở Task 04. |
| Kết quả & dự đoán từng mẫu 100% (ViSoBERT & PhoBERT) | `results/` (chỉ có `token_stats.csv` & `polarity/*`); ngoài repo: `null` | **MISSING** | Tìm thấy `0` CSV dự đoán test; chưa có `zs.csv`/`seq.csv`/predictions 100% local để tái tính view phụ & đối chứng mốc 100%. |
| Notebook Probe (Other/Surprise) | `null (không có .ipynb trong workspace)` | **MISSING** | Chưa có notebook Probe (Other/Surprise) trong workspace local hoặc chưa cấu hình `phase1_external.probe_notebook_path`. |

## 2. Trạng thái Git & Stash hiện có (Chỉ đọc)

- **Nhánh hiện tại:** `phase2-resource-transfer` @ `29564eb3a810c96336ab2f2cdc267a144dbc4e24`
- **Kế thừa từ `seq-core` (`62df4cd`):** `True`
- **Số stash đang lưu (không tự apply):** `4`
  - `stash@{0}: On seq-core: !!GitHub_Desktop<seq-core>`
  - `stash@{1}: On p1: !!GitHub_Desktop<p1>`
  - `stash@{2}: On main: !!GitHub_Desktop<main>`
  - `stash@{3}: On phuong: !!GitHub_Desktop<phuong>`

## 3. Khoảng trống mã nguồn cần bổ sung cho Giai đoạn 2

- Chưa có chế độ DFT khởi tạo trực tiếp từ pretrained HuggingFace ID (uitnlp/visobert, xlm-roberta-base) trong src/seq/trainer.py (hiện load_encoder_new_head chỉ nạp checkpoint local chứa thư mục encoder/).
- Chưa có giao thức C0 low-resource: C0N/C0V trong src/seq/stage2.py hiện nạp checkpoint nguồn 100% cùng miền, gây rò rỉ dữ liệu 100% đích nếu dùng cho 10%/25%/50% (cần chốt ở Task 04 và viết lại ở Task 05).
- Chưa có bộ sinh & nạp tập con phân tầng lồng nhau 5% ⊂ 10% ⊂ 25% ⊂ 50% theo ID cho UIT-VSMEC và NEU-ESC (Task 03).
- Chưa hỗ trợ tiêu chí chọn best checkpoint trên dev bằng Macro-F1 5 lớp (bỏ Surprise/Other) trong TorchTrainer (Task 04–05).
- Chưa có log chi phí huấn luyện đầy đủ (total_params, trainable_params qua requires_grad, peak VRAM, GPU-giờ, tên GPU) trong SEQ_COLUMNS / trainer (Task 05).
- Chưa có LLRD (hệ số 0.8/0.9) và LoRA (r=8/16, alpha=2r, query/value, full head; cần thêm peft hoặc module tương đương) (Task 10).
- Chưa có góc nhìn nhãn bổ sung V5-VSMEC (Macro-F1 5 lớp), V3-noSurprise, V3-NEU-noToxic trong src/zs/views.py và hỗ trợ Holm m=2 + CI tỷ số SD giữa seed (Task 12).

## 4. Chi tiết tài nguyên ngoài Git cần chép từ Drive/Colab

1. **10 checkpoint nguồn ViSoBERT Giai đoạn 1:**
   - 5 seed (`0..4`) UIT-VSMEC 7 lớp (`ViSoBERT_Stage1-*_s{0..4}_lr*` gồm `encoder/` + tokenizer + `head.pt` + `stage1_ckpt_hash.json`).
   - 5 seed (`0..4`) NEU-ESC 3 lớp (`ViSoBERT_NEU3-*_s{0..4}_lr*` gồm `encoder/` + tokenizer + `head.pt` + `neu3_ckpt_hash.json`).
2. **Kết quả & dự đoán từng mẫu test ở mức 100% (ViSoBERT và PhoBERT):**
   - `zs.csv`, `seq.csv`, `selection_seq.json`, `stage1.csv`, `neu3.csv` (kèm log cấu hình thật để xác minh `baseline_provenance`).
   - Thư mục `predictions_zs/` (`REFV2V`, `REFN2N`, `ZSV2N`, `ZSN2V`) và `predictions_seq/` (`SEQV2N`, `SEQN2V`, `C0N`, `C0V`) để tính lại `Macro-F1 5 lớp`, `V3-noSurprise`, `Strict V3`, `V3-NEU-noToxic` ở Task 12.
3. **Notebook Probe (Other/Surprise):** để sửa `PRIOR` từ `train` và bỏ fallback ngầm sang `uitnlp/visobert` trước khi đưa số liệu probe vào báo cáo.

## 5. Kết luận Task 01

- **Tài nguyên sẵn sàng dùng ngay tại local:** Toàn bộ dữ liệu sạch đóng băng (`data/clean/`), dữ liệu gốc 4 lớp NEU-ESC (`data/raw/`), toàn bộ code nền `seq-core` (`src/seq/`, `src/zs/`, `src/stats/`), và cấu hình Giai đoạn 1.
- **Tài nguyên đang `MISSING` ở local (cần chép trước Task 07/08/12, KHÔNG chặn Task 02–05):** 10 checkpoint nguồn ViSoBERT, bộ CSV kết quả + dự đoán từng mẫu mức 100%, và notebook Probe.
