# Điểm vào cho công việc mới

- Bước 2 zero-shot: `AGENT_zscdt_core.md`, cấu hình `configs/zs.yaml`.
- Bước 3 sequential: **`AGENT_step3_seq.md`**, cấu hình **`configs/seq.yaml`**;
  đọc quyết định trong `docs/seq_notes.md`.
- `AGENT_step3_seq (1).md` chỉ là đường dẫn chuyển tiếp.
- Các task NEU3/Stage1/M* còn lại là mô tả công việc lịch sử. Chúng có thể
  chứa patience2, phụ thuộc mã cũ hoặc giao thức v2; không dùng để ghi đè
  seq-v1. Muốn tái lập baseline lịch sử cần config/log thật của lượt đó.

Quy tắc bảo trì toàn repo nằm ở `AGENTS.md`. Khi đổi giao thức, cập nhật
config, tài liệu và test cùng nhau; không sửa âm thầm một file riêng.
