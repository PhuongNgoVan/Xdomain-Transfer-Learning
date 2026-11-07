# Báo cáo độ đồng thuận gán nhãn cực tính (Giai đoạn A – CV-P3)

## 1. Tổng quan độ đồng thuận (Cohen's kappa & chỉ số phụ)

| Tập mẫu | Số câu (n) | po | pe | Cohen's κ | 95% CI (Bootstrap) | Bootstrap bị bỏ | κ trọng số tuyến tính | Gwet AC1 | PABAK |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Toàn bộ (All) | 100 | 0.610000 | 0.338800 | 0.410163 | [0.273419, 0.539251] | 0/2000 (0.000000) | 0.405941 | 0.428739 | 0.415000 |
| Other | 50 | 0.660000 | 0.425200 | 0.408490 | [0.208443, 0.602601] | 0/2000 (0.000000) | 0.332524 | 0.526924 | 0.490000 |
| Surprise | 50 | 0.560000 | 0.342800 | 0.330493 | [0.148023, 0.508016] | 0/2000 (0.000000) | 0.361493 | 0.367907 | 0.340000 |

- **Tổng số câu bất đồng (`label_nhat != label_long`):** 39 / 100

## 2. Phân bố cực tính theo người gán và nhãn gốc (kèm 95% Wilson CI)

| Người gán | Nhãn gốc | Cực tính | Số đếm | Tổng | Tỉ lệ | Wilson 95% CI |
| --- | --- | --- | ---: | ---: | ---: | --- |
| nhat | All | negative (0) | 18 | 100 | 0.180000 | [0.117002, 0.266674] |
| nhat | All | neutral (1) | 38 | 100 | 0.380000 | [0.290976, 0.477902] |
| nhat | All | positive (2) | 44 | 100 | 0.440000 | [0.346720, 0.537719] |
| long | All | negative (0) | 26 | 100 | 0.260000 | [0.184047, 0.353710] |
| long | All | neutral (1) | 56 | 100 | 0.560000 | [0.462281, 0.653280] |
| long | All | positive (2) | 18 | 100 | 0.180000 | [0.117002, 0.266674] |
| nhat | Other | negative (0) | 12 | 50 | 0.240000 | [0.142974, 0.374127] |
| nhat | Other | neutral (1) | 27 | 50 | 0.540000 | [0.403989, 0.670303] |
| nhat | Other | positive (2) | 11 | 50 | 0.220000 | [0.127539, 0.352415] |
| long | Other | negative (0) | 17 | 50 | 0.340000 | [0.224369, 0.478462] |
| long | Other | neutral (1) | 31 | 50 | 0.620000 | [0.481504, 0.741372] |
| long | Other | positive (2) | 2 | 50 | 0.040000 | [0.011039, 0.134601] |
| nhat | Surprise | negative (0) | 6 | 50 | 0.120000 | [0.056176, 0.238048] |
| nhat | Surprise | neutral (1) | 11 | 50 | 0.220000 | [0.127539, 0.352415] |
| nhat | Surprise | positive (2) | 33 | 50 | 0.660000 | [0.521538, 0.775631] |
| long | Surprise | negative (0) | 9 | 50 | 0.180000 | [0.097702, 0.307961] |
| long | Surprise | neutral (1) | 25 | 50 | 0.500000 | [0.366445, 0.633555] |
| long | Surprise | positive (2) | 16 | 50 | 0.320000 | [0.207582, 0.458103] |

## 3. Cảnh báo

- không kiểm chứng được nguồn câu
- hai κ riêng nằm hai phía của 0.400000: Other κ=0.408490, Surprise κ=0.330493
- lệch phân bố cực tính giữa Nhật và Long trên nhóm All (max_diff=0.260000)
- kết luận yếu: Wilson 95% CI của cực lớn nhất (neutral) cho Other (nhat) chứa 0.60 [0.403989, 0.670303]
- cực lớn nhất có thể không chiếm đa số: Wilson 95% CI của cực lớn nhất (neutral) cho Other (nhat) chứa 0.50 [0.403989, 0.670303]
- kết luận yếu: Wilson 95% CI của cực lớn nhất (neutral) cho Other (long) chứa 0.60 [0.481504, 0.741372]
- cực lớn nhất có thể không chiếm đa số: Wilson 95% CI của cực lớn nhất (neutral) cho Other (long) chứa 0.50 [0.481504, 0.741372]
- lệch phân bố cực tính giữa Nhật và Long trên nhóm Other (max_diff=0.180000)
- kết luận yếu: Wilson 95% CI của cực lớn nhất (positive) cho Surprise (nhat) chứa 0.60 [0.521538, 0.775631]
- kết luận yếu: Wilson 95% CI của cực lớn nhất (neutral) cho Surprise (long) chứa 0.60 [0.366445, 0.633555]
- cực lớn nhất có thể không chiếm đa số: Wilson 95% CI của cực lớn nhất (neutral) cho Surprise (long) chứa 0.50 [0.366445, 0.633555]
- lệch phân bố cực tính giữa Nhật và Long trên nhóm Surprise (max_diff=0.340000)
