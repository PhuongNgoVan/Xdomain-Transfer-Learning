# Quy tắc chuyển 7 cảm xúc sang 3 cực tính (`W_trans` – CV-P3)

- **Ngày chốt (`--date`):** `2026-11-08`
- **Nguồn ngưỡng quyết định:** `mặc định`
- **Trạng thái phê duyệt (`wtrans_approval.md`):** `Chưa phê duyệt (dùng ngưỡng mặc định)`
- **Người phê duyệt ngưỡng:** `mặc định (chưa có wtrans_approval.md hợp lệ)`
- **Người chốt nhãn bất đồng (`final_label`):** `Nhật & Long (hoặc người thứ ba)`
- **Ngưỡng Cohen's κ toàn bộ (`kappa_threshold`):** `0.400000`
- **Ngưỡng tỉ lệ cực lớn nhất (`pole_threshold`):** `0.600000` (số đếm nguyên tối thiểu: `30/50`)

## 1. Độ đồng thuận trên nhãn độc lập (Giai đoạn A)

| Tập mẫu | Số câu (n) | po | pe | Cohen's κ | 95% CI (Bootstrap) | Bootstrap bị bỏ | κ trọng số tuyến tính | Gwet AC1 | PABAK |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Toàn bộ (All) | 100 | 0.610000 | 0.338800 | 0.410163 | [0.273419, 0.539251] | 0/2000 (0.000000) | 0.405941 | 0.428739 | 0.415000 |
| Other | 50 | 0.660000 | 0.425200 | 0.408490 | [0.208443, 0.602601] | 0/2000 (0.000000) | 0.332524 | 0.526924 | 0.490000 |
| Surprise | 50 | 0.560000 | 0.342800 | 0.330493 | [0.148023, 0.508016] | 0/2000 (0.000000) | 0.361493 | 0.367907 | 0.340000 |

## 2. Phân bố nhãn cuối (`final_label`) và luật chọn cho từng cột

| Cảm xúc | Số câu | Đếm (neg, neu, pos) | Tỉ lệ (neg, neu, pos) | Cực lớn nhất | Wilson 95% CI (cực lớn nhất) | Luật được chọn | Vectơ cột (neg, neu, pos) |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Surprise | 50 | (9, 25, 16) | (0.180000, 0.500000, 0.320000) | neutral (25/50) | [0.366445, 0.633555] | **SOFT** | `(0.180000, 0.500000, 0.320000)` |
| Other | 50 | (17, 31, 2) | (0.340000, 0.620000, 0.040000) | neutral (31/50) | [0.481504, 0.741372] | **HARD** | `(0.000000, 1.000000, 0.000000)` |

## 3. Ma trận `W_trans` (3 × 7, cột tổng bằng 1)

| label | Enjoyment | Sadness | Fear | Anger | Disgust | Surprise | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| negative | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.180000 | 0.000000 |
| neutral | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 1.000000 |
| positive | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.320000 | 0.000000 |

## 4. SHA-256 các file nhãn đầu vào

- `polarity_nhat.csv`: `a22f50bcd290569e65e943b2e2c1c575b176762fd1941560458841d237df98da`
- `polarity_long.csv`: `054ffc6808815a79ae9c3601d5e12977fbbadf6ba971d2e1e8ed17da2a05b0e6`
- `polarity_key.csv`: `4c5587afe006a905c71af1441b1125a0bcda1035786e66ec828a1a692201b15b`

## 5. Cảnh báo

- không kiểm chứng được nguồn câu
- đang dùng ngưỡng mặc định (kappa_threshold=0.40, pole_threshold=0.60) do chưa có wtrans_approval.md hợp lệ
- hai κ riêng nằm hai phía của 0.400000: Other κ=0.408490, Surprise κ=0.330493
- kết luận yếu (Surprise): Wilson 95% CI [0.366445, 0.633555] của cực lớn nhất (neutral) chứa ngưỡng 0.600000
- cực lớn nhất có thể không chiếm đa số (Surprise): Wilson 95% CI [0.366445, 0.633555] chứa 0.50
- kết luận yếu (Other): Wilson 95% CI [0.481504, 0.741372] của cực lớn nhất (neutral) chứa ngưỡng 0.600000
- cực lớn nhất có thể không chiếm đa số (Other): Wilson 95% CI [0.481504, 0.741372] chứa 0.50

## 6. Nội dung khai báo `WTRANS_INIT_RULE`

```python
# ---------------------------------------------------------------------------
# 5. W_trans Initialization Rule (7 emotions -> 3 sentiment probabilities)
#    Declared after G1 / kappa approval per M7 (CV-P3)
# ---------------------------------------------------------------------------
WTRANS_INIT_RULE: Dict[int, Tuple[float, float, float]] = {
    EMO_ENJOYMENT: (0.000000, 0.000000, 1.000000),
    EMO_SADNESS: (1.000000, 0.000000, 0.000000),
    EMO_FEAR: (1.000000, 0.000000, 0.000000),
    EMO_ANGER: (1.000000, 0.000000, 0.000000),
    EMO_DISGUST: (1.000000, 0.000000, 0.000000),
    EMO_SURPRISE: (0.180000, 0.500000, 0.320000),
    EMO_OTHER: (0.000000, 1.000000, 0.000000),
}
```
