# vi-emo2sent

Chuyển giao 7 cảm xúc (UIT-VSMEC) sang 3 nhãn quan điểm (NEU-ESC) và đánh giá ngoài miền trên UIT-VSFC.

## Cài đặt môi trường

Yêu cầu Python 3.10+. Khuyến nghị sử dụng môi trường ảo (`.venv`):

```bash
# Tạo và kích hoạt môi trường ảo (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Hoặc trên Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện đã khóa phiên bản
pip install -r requirements.txt
```

## Chạy kiểm thử (Unit Tests)

Toàn bộ unit test chạy hoàn toàn ngoại tuyến (không tải mạng) trên CPU:

```bash
pytest -v
```

## Tài liệu tham chiếu
- Quy tắc chung và phân công nhiệm vụ: [`AGENTS.md`](AGENTS.md)
- Hợp đồng giao diện (nguồn sự thật duy nhất): [`docs/contract.md`](docs/contract.md)
- Chi tiết các nhiệm vụ theo giai đoạn: [`tasks/`](tasks/)
