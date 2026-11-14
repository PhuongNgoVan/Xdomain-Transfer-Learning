# Báo cáo Kiểm tra Sẵn sàng Dữ liệu & Checkpoint cho Bước 2 (Zero-Shot VSMEC ↔ NEU-ESC)

**Kết luận cổng:** `CHƯA SẴN SÀNG: E1, E2, E3, F1, F3, G1, G2`

## 1. Bảng tổng hợp kết quả kiểm tra (A1 – J3)

| ID | Mức | Trạng thái | Chi tiết |
| --- | --- | --- | --- |
| `A1` | BLOCKER | **PASS** | Đủ 10 file mục 1, đọc được UTF-8, đúng cột quy định |
| `A2` | BLOCKER | **PASS** | VSMEC nhãn ∈ {0..6}, NEU nhãn ∈ {0,1,2}; id duy nhất; text & text_seg không rỗng/NaN |
| `A3` | BLOCKER | **PASS** | VSMEC=5542/686/692; NEU=23048/3305/6613 |
| `A4` | BLOCKER | **PASS** | EMOTIONS7 và (NEG, NEU, POS) == (0, 1, 2) khớp hằng số chuẩn |
| `B1` | BLOCKER | **PASS** | HARD_MERGE_RULE & HARD_MERGE_GROUPS khớp bảng v3.0 (0->2; 1,2,3,4->0; 5,6->1) |
| `B2` | BLOCKER | **PASS** | Hàm cộng xác suất 7->3 bảo toàn tổng hàng = 1.0, đúng thứ tự cột (neg, neu, pos) |
| `B3` | BLOCKER | **PASS** | NEUESC_ORIG_TO_LABEL3 khớp (toxic 3->neg 0); phân bố thực: neg=6045, neu=22773, pos=4148 |
| `C1` | BLOCKER | **PASS** | 0 câu trùng nội bộ trong cả 3 split của VSMEC và 3 split của NEU-ESC |
| `C2` | BLOCKER | **PASS** | 0 xung đột nhãn nội bộ và liên split trên cả VSMEC và NEU-ESC |
| `C3` | BLOCKER | **PASS** | train ∩ dev = 0, train ∩ test = 0, dev ∩ test = 0 cho cả VSMEC và NEU-ESC |
| `D1` | BLOCKER | **PASS** | 0 câu trong vsmec_test xuất hiện trong NEU-ESC (train, dev, test) |
| `D2` | BLOCKER | **PASS** | 0 câu trong neuesc_test xuất hiện trong VSMEC (train, dev, test) |
| `D3` | INFO | **PASS** | Đã ghi bảng đếm trùng chéo miền 3×2×2 (và ma trận 3×3) vào báo cáo |
| `D4` | WARN | **FAIL** | Giữ dữ liệu đóng băng; 2 cặp trùng gần: vsmec_test id=6311 (Sadness→negative) ↔ neuesc_dev id=[25960] (negative→negative, khớp=True); neuesc_test id=28482 (neutral→neutral) ↔ vsmec_train id=[3607] (Other→neutral, khớp=True). Ảnh hưởng độ nhạy: vsmec_test 1/692 (0.1445%), neuesc_test 1/6613 (0.0151%) — đã xuất sensitivity_exclude_ids.csv |
| `E1` | BLOCKER | **MISSING** | vsmec_strict_file=null trong configs/step2_data.yaml. Chưa có file VSMEC-strict trên vsmec_test (các file polarity_{nhat,long,key}.csv cũ lấy từ vsmec_train cho CV-P3 nên KHÔNG dùng được). Cần người phụ trách cung cấp file gán nhãn tay hoặc quyết định quy tắc strict |
| `E2` | BLOCKER | **MISSING** | vsmec_strict_file=null trong configs/step2_data.yaml. Chưa có file VSMEC-strict trên vsmec_test (các file polarity_{nhat,long,key}.csv cũ lấy từ vsmec_train cho CV-P3 nên KHÔNG dùng được). Cần người phụ trách cung cấp file gán nhãn tay hoặc quyết định quy tắc strict |
| `E3` | BLOCKER | **MISSING** | Chưa chốt quy tắc VSMEC-strict (bỏ các câu Surprise/Other hay gán lại bằng nhãn tay) và chưa có docs/vsmec_strict_rule.md |
| `F1` | BLOCKER | **MISSING** | Chưa có CSV kết quả Bước 1 trong reports/step2/inputs/ hoặc qua --results-file (hash hiện tại: vsmec_data_hash=680bfc0f241e..., neuesc_data_hash=9b0e3f42c2b7...) |
| `F2` | BLOCKER | **PASS** | Tất cả 8 file Bước 2 trong step2_files khớp SHA-256 trong MANIFEST.json |
| `F2b` | WARN | **FAIL** | Artifact cũ ngoài phạm vi Bước 2 (lệch MANIFEST đã biết, không dùng cho Bước 2): manifest_run1.json: thực=80a19181ad2c..., MANIFEST=không có; polarity_key.csv: thực=4c5587afe006..., MANIFEST=không có; polarity_long.csv: thực=054ffc680881..., MANIFEST=bba9a68c2af3...; polarity_nhat.csv: thực=a22f50bcd290..., MANIFEST=bba9a68c2af3...; vsfc_test_clean.csv: thực=e6873a782eb1..., MANIFEST=e6873a782eb1...; vsmec_clean_report.json: thực=7071ea3ec51d..., MANIFEST=không có |
| `F3` | BLOCKER | **MISSING** | Chưa có kết quả Bước 1 để kiểm tra tính duy nhất của data_hash và code_version |
| `G1` | BLOCKER | **MISSING** | Chưa có kết quả Bước 1 (tổng 0/20 lượt: VSMEC PhoBERT: 0/5; VSMEC ViSoBERT: 0/5; NEU-3 PhoBERT: 0/5; NEU-3 ViSoBERT: 0/5; cần người phụ trách tải stage1.csv và neu3.csv của 2 backbone vào reports/step2/inputs/) |
| `G2` | BLOCKER | **MISSING** | Thiếu đầu vào đối chiếu G2: vsmec_published_macro_f1.phobert=null, thiếu kết quả test VSMEC PhoBERT, vsmec_published_macro_f1.visobert=null, thiếu kết quả test VSMEC ViSoBERT |
| `G3` | WARN | **MISSING** | Chưa đủ kết quả để kiểm tra std và F1 từng lớp >= 0.30 (thiếu: VSMEC PhoBERT, VSMEC ViSoBERT, NEU-3 PhoBERT, NEU-3 ViSoBERT) |
| `H1` | INFO | **PASS** | vsmec_test 3c={'negative': 333, 'neutral': 166, 'positive': 193}; neuesc_test 3c={'negative': 1221, 'neutral': 4558, 'positive': 834} |
| `H2` | WARN | **PASS** | JS(3c train)=0.1633 bits (0.1132 nats), JS(3c test)=0.1524 bits; lớp nhỏ nhất vsmec_test 7c là Surprise=37 (>=30) |
| `H3` | INFO | **PASS** | Trong vsmec_test (n=692): Surprise=37 (5.35%), Other=129 (18.64%), tổng gộp vào neutral theo B1 = 166 (23.99%) |
| `H4` | INFO | **PASS** | Lệch bề mặt test: vsmec_test (' .'=6.65%, emoji=19.08%, words=13.80/p95=31.0, lower=100.00%) vs neuesc_test (' .'=98.53%, emoji=0.00%, words=26.02/p95=89.0, lower=100.00%) |
| `I1` | WARN | **PASS** | Cả vsmec_test (692 câu) và neuesc_test (6613 câu) đều có cột text_seg đầy đủ, không rỗng |
| `I2` | WARN | **PASS** | Cắt cụt @256: vsmec_test (PhoBERT=0/692 [0.00%], ViSoBERT=0/692 [0.00%]); neuesc_test (PhoBERT=45/6613 [0.68%], ViSoBERT=70/6613 [1.06%]) |
| `I3` | WARN | **PASS** | normalize_text(normalize_text(x)) == normalize_text(x) trên toàn bộ 7305 câu test |
| `I4` | WARN | **PASS** | ViSoBERT emoji_mode='keep' (preprocess_text(..., emoji_mode='keep') trên cột text) đồng nhất, idempotent và không sinh chuỗi rỗng trên cả vsmec_test và neuesc_test |
| `J1` | BLOCKER | **SKIP** | Bỏ qua kiểm tra checkpoint (chỉ chạy khi truyền --check-ckpt) |
| `J2` | BLOCKER | **SKIP** | Bỏ qua kiểm tra checkpoint (chỉ chạy khi truyền --check-ckpt) |
| `J3` | BLOCKER | **SKIP** | Bỏ qua kiểm tra checkpoint (chỉ chạy khi truyền --check-ckpt) |

## 2. Kiểm tra trùng chéo miền VSMEC ↔ NEU-ESC (D1 – D4)

### 2.1. Bảng đếm trùng chính xác (sau `normalize_text` + `hash_text`) theo 2 hướng (D3)

| Tập Test | Tập đối chiếu | Số hash trùng duy nhất | Số dòng ở Tập Test bị trùng | Số dòng ở Tập đối chiếu bị trùng |
| --- | --- | ---: | ---: | ---: |
| `vsmec_test` | `neuesc_train` | 0 | 0 | 0 |
| `vsmec_test` | `neuesc_dev` | 0 | 0 | 0 |
| `vsmec_test` | `neuesc_test` | 0 | 0 | 0 |
| `neuesc_test` | `vsmec_train` | 0 | 0 | 0 |
| `neuesc_test` | `vsmec_dev` | 0 | 0 | 0 |
| `neuesc_test` | `vsmec_test` | 0 | 0 | 0 |

### 2.2. Câu trùng gần phát hiện ở D4 (hạn chế đã biết & kiểm tra độ nhạy): `2` cặp

| Phía Test | ID | Nhãn gốc (7c/3c) | Nhãn B1 (3c) | Phía đối chiếu | ID khớp | Nhãn gốc đối chiếu | Nhãn B1 đối chiếu | Khớp nhãn 3c? | Chuỗi chuẩn hóa mạnh |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `vsmec_test` | 6311 | `Sadness` (1) | `negative` (0) | `neuesc_dev` | `[25960]` | `negative` (0) | `negative` (0) | **True** | `chạm đáy nỗi đau` |
| `neuesc_test` | 28482 | `neutral` (1) | `neutral` (1) | `vsmec_train` | `[3607]` | `Other` (6) | `neutral` (1) | **True** | `vật chất quyết định ý thức` |

- **Tỉ lệ câu bị ảnh hưởng nếu kiểm tra độ nhạy:** `vsmec_test` = **1/692** (0.1445%); `neuesc_test` = **1/6613** (0.0151%).
- **Quyết định đã chốt:** Giữ nguyên dữ liệu đóng băng, không loại bỏ câu nào khỏi tập test gốc. Hai ID vi phạm được xuất ra [sensitivity_exclude_ids.csv](sensitivity_exclude_ids.csv) để Bước 2 chạy thêm một lượt độ nhạy song song với lượt đầy đủ.

## 3. Phân bố nhãn, độ lệch tiên nghiệm và độ lệch bề mặt (H1 – H4)

### 3.1. Phân bố VSMEC 7 nhãn (H1)

| Split | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 1558 | 947 | 316 | 391 | 1069 | 241 | 1020 | **5542** |
| `dev` | 214 | 86 | 31 | 49 | 135 | 30 | 141 | **686** |
| `test` | 193 | 116 | 46 | 40 | 131 | 37 | 129 | **692** |

### 3.2. Phân bố 3 nhãn cực tính (VSMEC theo B1 vs NEU-ESC) và JS Divergence (H1, H2)

| Bộ dữ liệu | Split | negative (0) | neutral (1) | positive (2) | Tổng | JS Divergence (bits) | JS Divergence (nats) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VSMEC (gộp B1) | `train` | 2723 | 1261 | 1558 | **5542** | 0.163287 | 0.113182 |
| NEU-ESC | `train` | 4215 | 15936 | 2897 | **23048** | — | — |
| VSMEC (gộp B1) | `dev` | 301 | 171 | 214 | **686** | 0.145426 | 0.100802 |
| NEU-ESC | `dev` | 609 | 2279 | 417 | **3305** | — | — |
| VSMEC (gộp B1) | `test` | 333 | 166 | 193 | **692** | 0.152440 | 0.105664 |
| NEU-ESC | `test` | 1221 | 4558 | 834 | **6613** | — | — |

### 3.3. Hệ quả của B1 lên Surprise và Other trong `vsmec_test` (H3)

- Tổng số câu `vsmec_test`: **692**
- `Surprise` (nhãn 5): **37** câu (5.35%)
- `Other` (nhãn 6): **129** câu (18.64%)
- Tổng gộp vào `neutral` (nhãn 1) theo quy tắc cứng B1: **166** câu (23.99%)

### 3.4. Lệch bề mặt giữa hai miền (H4)

| Bộ dữ liệu | Split | Số câu | Kết thúc `" ."` | Kết thúc `"."` | Có emoji Unicode | Số từ TB | Số từ p95 | Toàn chữ thường |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VSMEC | `train` | 5542 | 6.42% | 8.88% | 17.27% | 14.01 | 33.0 | 100.00% |
| VSMEC | `dev` | 686 | 5.39% | 7.87% | 16.62% | 14.18 | 31.0 | 100.00% |
| VSMEC | `test` | 692 | 6.65% | 9.10% | 19.08% | 13.80 | 31.0 | 100.00% |
| NEU-ESC | `train` | 23048 | 98.66% | 98.66% | 0.00% | 24.81 | 81.0 | 99.99% |
| NEU-ESC | `dev` | 3305 | 98.70% | 98.70% | 0.00% | 25.12 | 82.0 | 99.97% |
| NEU-ESC | `test` | 6613 | 98.53% | 98.53% | 0.00% | 26.02 | 89.0 | 100.00% |

Hiệu ứng này cần được xét ở phần "độ lệch nội tại" của Bước 2.

## 4. Artifact cũ ngoài phạm vi Bước 2 trong `data/clean/` (F2b)

> Các file dưới đây là artifact cũ (CV-P3 / mở rộng), **không** được Bước 2 sử dụng. Sai khác với `MANIFEST.json` là đã biết và giữ nguyên `MANIFEST.json` theo quyết định.

| File | SHA-256 thực tế | SHA-256 trong `MANIFEST.json` | Ghi chú |
| --- | --- | --- | --- |
| `manifest_run1.json` | `80a19181ad2c88373af0b93289c978328bed4569963acbb4a5a5a8d3d2fdde9c` | không có | không có trong MANIFEST |
| `polarity_key.csv` | `4c5587afe006a905c71af1441b1125a0bcda1035786e66ec828a1a692201b15b` | không có | không có trong MANIFEST |
| `polarity_long.csv` | `054ffc6808815a79ae9c3601d5e12977fbbadf6ba971d2e1e8ed17da2a05b0e6` | `bba9a68c2af398bdca44eccf6fea1194802f26b641fa1455c8ddbe7eddfc6834` | lệch MANIFEST |
| `polarity_nhat.csv` | `a22f50bcd290569e65e943b2e2c1c575b176762fd1941560458841d237df98da` | `bba9a68c2af398bdca44eccf6fea1194802f26b641fa1455c8ddbe7eddfc6834` | lệch MANIFEST |
| `vsfc_test_clean.csv` | `e6873a782eb1368b85a46fcabee091aa081a174f01fdc759951558aba4ed6023` | `e6873a782eb1368b85a46fcabee091aa081a174f01fdc759951558aba4ed6023` | khớp MANIFEST |
| `vsmec_clean_report.json` | `7071ea3ec51d3a8a72e597e39225757b139a33003758e6f52400b94dd4b59dd8` | không có | không có trong MANIFEST |

## 5. Đoạn mã Colab kiểm tra checkpoint & tokenizer trên Drive (nhóm I2 & J1–J3)

```python
from google.colab import drive
drive.mount('/content/drive')

# Chỉnh lại đường dẫn repo & thư mục Drive thực tế của bạn:
%cd /content/Xdomain-Transfer-Learning
!git checkout p1

!python -m scripts.check_step2_data \
  --results-glob '/content/drive/MyDrive/xdomain_results/**/*.csv' \
  --with-tokenizer \
  --check-ckpt \
  --reload-check \
  --ckpt-root-vsmec-phobert '/content/drive/MyDrive/xdomain_ckpts/stage1' \
  --ckpt-root-vsmec-visobert '/content/drive/MyDrive/xdomain_ckpts/stage1' \
  --ckpt-root-neu3-phobert '/content/drive/MyDrive/xdomain_ckpts/neu3' \
  --ckpt-root-neu3-visobert '/content/drive/MyDrive/xdomain_ckpts/neu3'
```
