# Định nghĩa thống kê seq-v1

Gain=mean_seed(F1_seq)−mean_seed(F1_base), điểm trên test đầy đủ, không lấy
trung bình bootstrap. Baseline: REFN2N cho NEU, REFV2V cho VSMEC, cùng ID,
nhãn và thứ tự seed. Macro-F1 labels: V3 {0,1,2}, V7 {0..6}, strict {0,2};
zero_division=0. Neutral dự đoán trên strict vẫn sai, không xuất F1 neu.

Bootstrap B10000, PCG64(seed0), chunk500, percentile CI95%. Hai arm dùng
chung trọng số câu và chỉ số seed rút lại; phải có seed ID duy nhất cùng
thứ tự. C0/S−C0 dùng giao seed. Zero-shot mặc định share_seeds=False,
giữ nguyên chuỗi ngẫu nhiên cũ.

p_boot=min(1,2 min((count(delta<=0)+1)/(B+1),
(count(delta>=0)+1)/(B+1))). Sàn=2/(B+1). Đây là thống kê đuôi bootstrap
không tái tâm theo giao thức chọn, không phải p-value hoán vị dưới H0;
không khẳng định hiệu chuẩn sai số loại I từ công thức này. Với năm seed
có 126 multiset; cả tầng seed lẫn tầng câu đều có thể chi phối phương sai.
Báo CI, std giữa seed và số seed Gain dương; không kết luận từ p sát0.05
hoặc chênh lệch nhỏ trên VSMEC692 câu.

Holm chỉ cho bốn primary macro-F1 test, m=4: cực đại tích lũy
min(1,(4−i+1)*p_i) theo p tăng rồi hoàn về thứ tự gốc; significant=p_adj<0.05.
Chỉ điền khi đủ bốn so sánh và năm seed ghép mỗi so sánh. Thiếu thì để
trống, không tự thu nhỏ gia đình. Mọi phân tích khác exploratory=True.

Gain_C0=C0−base, S−C0 trên giao seed. Ma trận cộng qua seed rồi chuẩn hóa
hàng. seq_vs_zs chỉ ghép view3 cùng không gian nhãn; không suy nhân quả
từ bốn điểm. Provenance chưa xác minh phải hiện trong metadata/báo cáo.
