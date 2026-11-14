# vi-emo2sent

Nghiên cứu chuyển giao hai chiều UIT-VSMEC ↔ NEU-ESC theo workflow v3.0:
baseline, zero-shot và sequential transfer. UIT-VSFC nằm ngoài lõi hiện hành.

Sequential dùng [configs/seq.yaml](configs/seq.yaml) làm cấu hình duy nhất;
đọc [phương án đã chốt](docs/seq_notes.md) và
[hợp đồng triển khai](tasks/AGENT_step3_seq.md). Trainer độc lập ở src/seq;
không cần copy các file train cũ từ nhánh khác. Tài liệu v2.x chỉ để tra cứu.

Quy trình: sweep train/dev → chọn LR → main/C0 → phân tích cùng prediction
tham chiếu Bước 2. Không chấm test trong sweep. Phải có config/log baseline
thật trước khi coi so sánh là tương đương về giao thức.

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



