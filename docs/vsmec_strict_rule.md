# Quy tắc VSMEC-strict

Theo kế hoạch nghiên cứu v3.0 mục 3.2, dựng `V3-VSMEC-strict` bằng cách
lọc `data/clean/vsmec_test.csv`, chỉ giữ nhãn gốc `label <= 4`:
Enjoyment (0), Sadness (1), Fear (2), Anger (3), Disgust (4).
Bỏ Surprise (5, 37 câu) và Other (6, 129 câu), còn **526/692 câu**.
Giữ nguyên ID, thứ tự và nội dung; chỉ lọc khi đánh giá, không sửa dữ liệu đóng băng.

Ánh xạ nhãn bằng `HARD_MERGE_RULE`: Enjoyment → positive (2), bốn cảm xúc
còn lại → negative (0). Macro-F1 tính trên **hai lớp `{0, 2}`**, với
`zero_division=0`; dự đoán neutral (1) vẫn tính là sai. `zs_per_class.csv`
chỉ xuất `neg` và `pos` cho view này.

Cấu hình readiness: `vsmec_strict_rule: label_le_4`; không cần CSV gán nhãn
tay (`vsmec_strict_file: null`). E1 kiểm tra dữ liệu nguồn và tập con lọc;
E2 không áp dụng kiểm tra hai người gán/Cohen's kappa (`SKIP`);
E3 kiểm tra tài liệu quy tắc và tập strict không rỗng. Kiểm chứng thủ công
Other/Surprise là phân tích riêng, không dùng để dựng lại tập strict này.
