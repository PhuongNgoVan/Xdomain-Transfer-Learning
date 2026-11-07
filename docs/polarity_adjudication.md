# Chốt nhãn bất đồng polarity (CV-P3)
- Ngày quyết định: 2026-11-08 (sau khi đã xem kappa_report.md của Giai đoạn A)
- Người chốt: Long. final_label = label_long cho 39/100 câu bất đồng.
  Hệ quả: nhãn cuối trùng toàn bộ nhãn của Long.
- Lý do: Long điền theo quy ước, Nhật điền không có quy ước
- Hạn chế đã biết: κ = 0.410 (CI [0.273, 0.539]), bất đồng 39%.
  Dùng nhãn Nhật thì Surprise thành HARD positive (33/50) và Other thành SOFT.
  Kết quả W_trans nhạy với người chốt.