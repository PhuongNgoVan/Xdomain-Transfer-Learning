# Task 05 — Pipeline nền phase2-v1

Ngày triển khai: 2026-11-26. Nhánh: `phase2-resource-transfer`.

## Phạm vi đã triển khai

- `src/phase2/data.py`: raw `text`, emoji keep, kiểm định underscore,
  tập con Task03, fingerprint văn bản và đánh dấu mọi lượt cắt `max_examples`.
- `src/phase2/models.py`: DFT pretrained, Seq encoder nguồn, C0 encoder từ
  best DFT cùng hướng/fraction/seed/ID. Kiểm tra artifact, cấu hình, hash và
  nguồn dữ liệu; head mới có cùng seed với Seq. Provenance lịch sử nguồn Seq
  vẫn `unverified`, không suy ra xác minh lượt thật từ checkpoint CPU.
- `src/phase2/trainer.py`: AdamW chỉ lấy tham số trainable, warmup tuyến tính,
  chọn best bằng dev V5/V3, early stopping, khôi phục best trước suy luận.
  Resume nguyên tử sau mỗi update, gồm cursor, best state, optimizer,
  scheduler, history và RNG Python/NumPy/Torch/CUDA. Mỗi work directory mới
  tạo optimizer/scheduler mới; resume từ thư mục cũ kiểm tra train/dev/settings.
- `src/phase2/run.py`: xuất prediction theo schema ZS, chấm các view từ CSV
  đã lưu, upsert bảng tổng hợp, completion có hash và kiểm tra fingerprint.
  Thiếu/hỏng artifact hoặc dòng tổng hợp thì tái tạo; fingerprint khác thì lỗi.
- `src/phase2/cli.py`, `select_lr.py`: CLI và chọn LR từ đủ 12 sweep dev-only,
  hòa chọn LR nhỏ hơn, cảnh báo mép lưới. Kiểm tra hash artifact, training
  profile, code, dữ liệu và protocol trước khi xuất selection.
- `configs/phase2_smoke_example.yaml`: ví dụ Task06 SA/DFT/10%/seed0/LR3e-5.

## Cách ly test và tương thích loader Task03

`load_phase2_target_splits(..., include_test=False)` là phần mở rộng trong
`src/phase2/subsets.py`; mặc định `True` giữ hành vi cũ. Sweep/smoke không
mở `vsmec_test.csv`, không chuẩn bị văn bản/nhãn test và không đưa test vào
trainer. NEU có chung file `neuesc_100.csv`: lọc train/dev trước kiểm định và
tiền xử lý; hash byte toàn file chỉ dùng định danh, theo quy ước seq-v1.

Với VSMEC dev-only, hash test lấy từ MANIFEST Task03 đã khóa; không tuyên bố
đã kiểm tra nội dung test trong lượt đó. Kiểm tra hash/nội dung/chống trùng
với test được thực hiện trong main. Provenance ghi `test_integrity_checked`.
Không thay đổi dữ liệu hay MANIFEST để phục vụ cách ly này.

## Giao diện Task10

Truyền `optimizer_builder(model, settings)`, `model_adapter_fn(model, cfg)`
và `checkpoint_saver(model, tokenizer, path)` vào `run_phase2_experiment`.
Trainer giữ tỷ lệ LR giữa các nhóm, clip chỉ tham số trainable, lưu/khôi phục
optimizer/scheduler tùy biến. Completion kiểm tra cả checkpoint tùy biến;
không buộc saver adapter phải tạo thư mục encoder đầy đủ.
`epochs_override=15` là điểm tích hợp cho quy tắc LoRA đã khóa; Task05 chưa
triển khai LLRD/LoRA hay tự quyết định mở rộng epoch.

`run_overrides` chứa thay đổi training settings có chủ đích; settings, hooks,
run mode và max_examples tham gia hash run ID. Fingerprint còn chứa dữ liệu,
nguồn checkpoint, Git commit và SHA-256 các file `src/phase2/*.py`.

## Cách gọi sau bàn giao

Chỉ chạy GPU Task06 khi được yêu cầu riêng:

```powershell
.venv/Scripts/python.exe -X utf8 -m src.phase2.cli --config configs/phase2_smoke_example.yaml
```

Các cờ `--pretrained-model-path`, `--source-ckpt-root`, `--c0-stage1-run-dir`
cho phép chỉ định nguồn. Run chính dùng `--selection-path` để xác minh LR đã
khóa; C0 lấy khóa `seq` cùng hướng. Nếu không truyền selection, API không
tự tìm hay giả định đã có LR được khóa; người chạy nghiên cứu phải truyền
selection của Task07. Smoke ví dụ vẫn giữ BF16 strict, không fallback ngầm.

## Kiểm chứng

- `py_compile` toàn bộ 6 module mới và test: PASS.
- Hai CLI `--help`: PASS, không launch training.
- Task03–04 hồi quy: 8 PASS.
- Task05: 15 PASS (11 nhóm yêu cầu, tham số hóa cả hướng và mode).
- Test Task05 dùng synthetic corpus, BERT 1 tầng cục bộ, FakeTrainer, CPU;
  kiểm tra resume giữa epoch/cuối epoch, best restore, head Seq/C0 từng bit,
  cách ly test, xác suất, view strict, hooks và artifact SKIP.
- Fixture hash trước/sau xác nhận tài nguyên đóng băng không đổi.
- Toàn suite trên phiên bản cuối: **98 PASS trong 129,75 giây** (83 test cũ
  và 15 trường hợp Task05). Lệnh `.venv/Scripts/python.exe -X utf8 -m pytest -q`;
  lượt kiểm tra cuối đặt `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` cho CPU.
- `git diff --check`: PASS. Diff các tài nguyên đóng băng hoàn toàn trống.

Task05 hoàn thành phần triển khai và kiểm thử offline, sẵn sàng bàn giao
Task06 và tích hợp Task10 trong phạm vi đã khóa.

Chưa chạy GPU nghiên cứu, truy cập Drive, tải mô hình từ mạng, xác minh
baseline lịch sử, commit hoặc push. `.gitignore` hiện ẩn file test Phase2;
khi chuẩn bị commit, người điều phối cần whitelist `!tests/test_phase2_*.py`
và `!tasks/AGENT_phase2_*.md` theo task.
