# Định nghĩa chỉ số zero-shot v3.0

## Góc nhìn nhãn và điểm ước lượng

Nhãn chung theo thứ tự negative (0), neutral (1), positive (2).
VSMEC 7 lớp giữ thứ tự từ `EMOTIONS7`; ánh xạ dùng hằng số frozen chung.
`sum` cộng xác suất softmax 7 lớp theo nhóm rồi argmax; `argmax` chọn
cảm xúc trước rồi ánh xạ. Đầu NEU-3 dùng `map_method=none`.

Macro-F1 dùng `labels=[0,1,2]` cho V3 đầy đủ, `[0,2]` cho strict,
`[0..6]` cho V7; `zero_division=0`. Weighted-F1 lấy trọng số hỗ trợ của
những nhãn tương ứng. Accuracy tính trên mọi câu của view, nên neutral
dự đoán trên strict vẫn sai. `f1_per_class` luôn dài 3 hoặc 7; vị trí
neutral không áp dụng ở strict là `null`, các cột scalar tương ứng trống.
Bảng `zs_per_class.csv` strict chỉ có `neg` và `pos`.

Điểm ước lượng của một arm là trung bình **điểm từng seed trên toàn bộ
test**, không tính trên dự đoán ensemble và không lấy trung bình bootstrap.
Báo thêm độ lệch chuẩn mẫu (`ddof=1`) giữa các seed (một seed ghi 0).

## Tham chiếu, gap và retention

- A, `V3-NEU`: ZS `zsA/sum`, REF `refN/none`.
- B, `V3-VSMEC`: ZS `zsB/none`, REF `refV/sum`.
- B, `V3-VSMEC-strict`: cùng hai arm trên cùng tập strict.

Gap = F1_REF − F1_ZS, đơn vị điểm macro-F1 trên thang 0..1.
Retention = F1_ZS / F1_REF, là tỷ số trung bình các seed, không lấy trung
bình các tỷ số ghép seed. Mẫu bootstrap có REF tuyệt đối nhỏ hơn `1e-8`
cho retention NaN, đếm riêng `n_nan` và bỏ khỏi phân vị/SE.

Bất đối xứng chiều = retention_A − retention_B (hoặc gap_A − gap_B)
cho từng backbone; B được báo cả full và strict. Bất đối xứng backbone =
PhoBERT − ViSoBERT cho từng chiều/view. Contrast được tính trong từng
lần bootstrap, không trừ cận CI rời nhau.

## JS, ma trận và mốc sàn

JS divergence log cơ số 2 của phân bố dự đoán và thật, trong [0,1],
giá trị bằng 0 khi trùng và bằng 1 khi tách rời. JS/ma trận 3×3 strict
vẫn giữ **cả ba lớp**, đặc biệt khối lượng dự đoán neutral không bị loại
hoặc chuẩn hóa mất. JS prior giữa hai miền dùng nhãn train ánh xạ 3 lớp.

Ma trận đếm cộng qua mọi seed, sau đó chuẩn hóa từng hàng; hàng tổng 0
giữ 0. Ma trận chiều B 7×3: cảm xúc VSMEC thật × nhãn NEU dự đoán.
Ma trận chiều A 3×7: nhãn NEU thật × argmax cảm xúc dự đoán. Các ma trận
không có bootstrap CI theo hợp đồng bảng; F1/JS/gap/retention có CI.

Majority-class dùng lớp phổ biến nhất trong train của miền test, không
dùng lớp phổ biến nhất test. Ngẫu nhiên prior dùng nhãn train cùng miền,
trung bình macro-F1 2.000 lượt mô phỏng PCG64 có seed cố định. Đây chỉ
là mốc báo cáo, không dùng để điều chỉnh suy luận zero-shot.

## Bootstrap hai tầng

Mặc định `B=10000`, CI percentile 95%, `PCG64(boot_seed)`, chunk 500.
Mỗi lần lặp rút n câu test có hoàn lại (trọng số multinomial), dùng cùng
trọng số câu cho mọi seed trong arm, rồi rút S seed có hoàn lại.
Thống kê là trung bình điểm của các seed đã rút trên test đã rút.

Các arm cùng test dùng chung trọng số câu, với ID/nhãn cùng thứ tự;
seed luôn rút độc lập giữa các arm. Hai chiều A/B khác test thì trọng số
câu độc lập. Các tỷ số/hiệu/bất đối xứng tính từ arm trong từng lần lặp.
Trả điểm toàn dữ liệu, phân vị 2,5/97,5, SE mẫu của bootstrap và số NaN.
Không p-value, không kiểm định Holm ở Bước 2; chúng thuộc Bước 3.

Chỉ có năm seed nên ước lượng phương sai giữa seed còn thô; luôn báo
std giữa seed. Test VSMEC 692 câu, strict 526 câu: khoảng tin cậy có thể
rộng, không kết luận từ chênh lệch vài điểm. Bootstrap không phân tầng
có thể bỏ mất lớp hiếm; theo `zero_division=0`, macro-F1 của classifier
hoàn hảo vẫn có thể giảm ở mẫu bootstrap thiếu lớp. Không thay thế giao
thức bằng bootstrap phân tầng chỉ để làm CI suy biến.
