# Task06 — Smoke DFT / NEU-ESC 10% / A100

Lượt Colab ngày **27/11/2026, giờ Việt Nam** (log ghi UTC ngày 26/11).
Commit: `af12db20672b198accf724cbd26cdbf59b32d162`.
Run ID: `ViSoBERT_SA_dft_f10_s0_lr3e-05_p2v1_h9c9d1961`.

## Đọc theo thứ tự

1. [Báo cáo smoke](smoke_report.md): cấu hình, kết quả và giới hạn phép đo.
2. [Bảng kết quả](artifacts/phase2_runs.csv): một dòng dev, không có test.
3. [Log train/resume](logs/train.log): lần ngắt có chủ đích và lần chạy tiếp.
4. [Môi trường Colab](environment/environment.json), [package](environment/pip-freeze.txt).

## Kết quả chính

| Mục | Giá trị |
|---|---|
| Phương pháp | ViSoBERT / DFT / SA / fraction 10 / seed 0 / LR 3e-5 |
| Dữ liệu | Train 2.305, dev 3.305; không chấm test |
| Best dev Macro-F1 | 0,6637240822, epoch 2, step 146 |
| Dừng sớm | Epoch 4, tổng 292 update |
| GPU / precision | NVIDIA A100-SXM4-80GB / BF16 strict |
| Thời gian trainer | 46,45 giây; không phải tổng thời gian phiên Colab |
| Peak VRAM | 2.979,66 MiB, khoảng 2,91 GiB |

Lượt smoke đã hoàn thành: artifact hợp lệ, thử ngắt/resume thành công theo
log, gọi lại trả SKIP. Đây chưa phải kết quả test hoặc LR được chọn từ sweep.
Torch Colab thực tế là 2.14.1+cu130, khác pin 2.9.0 trong repo.

## Cấu trúc lưu trữ

- `smoke_report.md`: báo cáo gốc, đổi tên từ `reports/phase2/smoke_test.md`.
- `artifacts/runs/<run_id>/`: config, metadata, metrics, cost, prediction dev,
  completion; giữ nguyên tên và byte để kiểm tra hash tương đối.
- `artifacts/work/<run_id>/provenance.json`: fingerprint nguồn lượt chạy.
- `logs/`: train/install/SKIP và biên bản thử resume.
- `environment/`: môi trường, package, hash tài nguyên trước khi chạy.
- `archive_index.json`: đường dẫn cũ → mới và SHA-256 của toàn bộ 16 file gốc.

Đây là **bản lưu để đọc/đối chiếu**, không tự sửa các đường dẫn `/content/...`
trong provenance/log gốc. Bản tải về không chứa `resume.pt` hoặc checkpoint
main encoder/head; không dùng thư mục này làm nguồn checkpoint C0 hoặc để
resume trực tiếp. Các hash byte của source/frozen từ Colab cần đối chiếu với
Git blob/LF khi đọc trên Windows CRLF. Digest dòng bảng CSV cũng dùng LF.

Ngày sắp xếp: 27/11/2026. Chỉ di chuyển/đổi tên file, xác minh SHA-256 trước
và sau; không thay đổi nội dung kết quả, không chạy thêm thực nghiệm.
