# Báo cáo Kiểm tra Sẵn sàng Dữ liệu & Checkpoint cho Bước 2 (Zero-Shot VSMEC ↔ NEU-ESC)

**Kết luận cổng:** `CHƯA SẴN SÀNG: E1, E2, E3, F1, F2, F3, G1, G2`

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
| `D4` | WARN | **FAIL** | Phát hiện 2 câu test trùng gần khi bỏ dấu câu/emoji (vsmec_test id=6311 ↔ neuesc_dev id=25960; neuesc_test id=28482 ↔ vsmec_train id=3607; xem violations_D4_near_duplicates.csv) |
| `E1` | BLOCKER | **MISSING** | vsmec_strict_file=null trong configs/step2_data.yaml. Cần người phụ trách cung cấp đường dẫn & định dạng file gán nhãn tay VSMEC-strict trên vsmec_test (lưu ý: polarity_{nhat,long,key}.csv hiện có trong data/clean/ được lấy mẫu từ vsmec_train cho W_trans ở CV-P3) |
| `E2` | BLOCKER | **MISSING** | vsmec_strict_file=null trong configs/step2_data.yaml. Cần người phụ trách cung cấp đường dẫn & định dạng file gán nhãn tay VSMEC-strict trên vsmec_test (lưu ý: polarity_{nhat,long,key}.csv hiện có trong data/clean/ được lấy mẫu từ vsmec_train cho W_trans ở CV-P3) |
| `E3` | BLOCKER | **MISSING** | vsmec_strict_file=null và chưa có tài liệu quy tắc dựng tập VSMEC-strict trong docs/ |
| `F1` | BLOCKER | **MISSING** | Chưa có file kết quả Bước 1 tại results/stage1.csv, results/neu3.csv trên nhánh hiện tại (hash hiện tại: vsmec_data_hash=680bfc0f241e..., neuesc_data_hash=9b0e3f42c2b7...); cần người phụ trách đồng bộ results/stage1.csv và results/neu3.csv từ Drive |
| `F2` | BLOCKER | **FAIL** | Lệch MANIFEST.json: polarity_long.csv (MANIFEST=bba9a68c2a... != thực tế=054ffc6808...); polarity_nhat.csv (MANIFEST=bba9a68c2a... != thực tế=a22f50bcd2...); CSV chưa có trong MANIFEST.json: ['polarity_key.csv'] |
| `F3` | BLOCKER | **MISSING** | Chưa có results/stage1.csv, results/neu3.csv để kiểm tra tính duy nhất của data_hash và code_version |
| `G1` | BLOCKER | **MISSING** | Chưa có results/stage1.csv, results/neu3.csv; cần người phụ trách cung cấp kết quả 20 lượt baseline Bước 1 (5 seed × 2 backbone × 2 miền) |
| `G2` | BLOCKER | **MISSING** | vsmec_published_macro_f1=null trong configs/step2_data.yaml; cần người phụ trách cung cấp số macro-F1 test VSMEC đã công bố để đối chiếu |
| `G3` | WARN | **MISSING** | Chưa có đủ results/stage1.csv và results/neu3.csv để kiểm tra std giữa các seed và F1 từng lớp >= 0.30 |
| `H1` | INFO | **PASS** | vsmec_test 3c={'negative': 333, 'neutral': 166, 'positive': 193}; neuesc_test 3c={'negative': 1221, 'neutral': 4558, 'positive': 834} |
| `H2` | WARN | **PASS** | JS(3c train)=0.1633 bits (0.1132 nats), JS(3c test)=0.1524 bits; lớp nhỏ nhất vsmec_test 7c là Surprise=37 (>=30) |
| `H3` | INFO | **PASS** | Trong vsmec_test (n=692): Surprise=37 (5.35%), Other=129 (18.64%), tổng gộp vào neutral theo B1 = 166 (23.99%) |
| `I1` | WARN | **PASS** | Cả vsmec_test (692 câu) và neuesc_test (6613 câu) đều có cột text_seg đầy đủ, không rỗng |
| `I2` | WARN | **SKIP** | Bỏ qua nạp tokenizer (truyền --with-tokenizer để đo tỉ lệ cắt cụt ở max_length=256) |
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

### 2.2. Câu trùng gần phát hiện ở D4 (bỏ dấu câu & emoji, giữ dấu tiếng Việt): `2` câu

| Tập Test | ID | Nhãn | Tập khớp | ID khớp | Chuỗi chuẩn hóa mạnh | Văn bản gốc |
| --- | ---: | ---: | --- | --- | --- | --- |
| `vsmec_test` | 6311 | 1 | `neuesc_dev` | `[25960]` | `chạm đáy nỗi đau` | chạm đáy nỗi đau 😞 |
| `neuesc_test` | 28482 | 1 | `vsmec_train` | `[3607]` | `vật chất quyết định ý thức` | vật chất quyết định ý thức . |

> **Lưu ý:** Nếu loại bỏ bất kỳ câu nào khỏi tập test (`vsmec_test` hoặc `neuesc_test`), số liệu test của Bước 1 bắt buộc phải được suy luận và tính lại từ checkpoint đã lưu.

## 3. Phân bố nhãn và độ lệch tiên nghiệm (H1 – H3)

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

## 4. Thông tin VSMEC-strict (E1–E3) và Cổng Bước 1 (G1–G3)

- **VSMEC-strict Cohen's kappa (E2):** `{}` (ghi chú: trên nhánh hiện tại có `results/polarity/kappa_report.json` cho 100 câu lấy từ `vsmec_train` ở CV-P3 với κ = 0.410163, nhưng chưa có file `vsmec_strict_file` trên `vsmec_test`)
- **Đối chiếu F1 công bố (G2):** `{}`
