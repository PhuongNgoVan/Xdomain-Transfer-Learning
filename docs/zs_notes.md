# Kiểm chứng nhánh zscdt-core

## Đầu vào và phạm vi

Nhánh thực tế xuất phát từ `p1` (commit `85c3627`), chưa có các module train
`src/plm/stage1.py` hoặc `src/plm/train_neu3.py`. Bộ nạp
`src/plm/model_ckpt.py` ban đầu **không tồn tại**, không phải một bản chỉ nạp
7 lớp. Bước chuẩn bị đã chép nguyên file từ `phobert-vsmec-7c` và kiểm tra
lưu/nạp checkpoint offline cho hai kích thước đầu 3/7 và hai alias. Bộ nạp
suy ra `num_labels` từ `head.pt`; giữ `accepts_loss_kwargs = False`.
Không mang file `train_neu3.py` bị self-import vào nhánh này.

Theo kế hoạch nghiên cứu/Workflow v3.0 trong `docs`, mã mới chỉ suy luận
zero-shot và tham chiếu trong miền, cộng xác suất/argmax rồi ánh xạ, báo cáo
bootstrap. Không train, không chỉnh ngưỡng, không oracle prior correction,
không PAD/UMAP hay phân tích theo nhóm câu. Không truy cập Drive hoặc chạy
checkpoint nghiên cứu thật trong đợt triển khai này.

## Checkpoint

- PhoBERT VSMEC có thể dùng `PhoBERT_Stage1-bf16_s0_lr2e-5`;
  locator còn nhận tùy chọn `base-` và `emokeep-` trước `bf16`.
- NEU-3 dùng `{alias}_NEU3-base[-emokeep]-bf16_s{seed}_lr{lr}`.
  Chấp nhận checkpoint trực tiếp trong thư mục run hoặc thư mục alias con.
- Bỏ mọi đường dẫn có thành phần `sweep`/`smoke`; thiếu hoặc nhiều hơn một
  baseline phù hợp thì lỗi, không chọn file đầu tiên.
- Trước khi nạp encoder, kiểm tra `config.json`, file trọng số không rỗng
  (`model.safetensors` hoặc `pytorch_model.bin`), tensor weight/bias của
  `head.pt` và kích thước đầu 7 cho VSMEC, 3 cho NEU-3.
- Khi có manifest hash `stage1_ckpt_hash.json`/`neu3_ckpt_hash.json` trong
  run/root hoặc thư mục results liên quan, phải có entry đúng run và đủ hash
  encoder/head khớp byte thật. Manifest rỗng/thiếu entry không được PASS.
  Không có manifest thì vẫn có thể chạy sau kiểm tra cấu trúc; cần giữ các
  file hash Bước 1 khi chuẩn bị Drive để kiểm chứng đầy đủ.
  Khi chuyển Colab → Drive, entry cũ khóa bằng alias được chấp nhận nếu
  manifest nằm bên trong chính thư mục run được chọn; hai hash vẫn phải
  khớp. Manifest chung chỉ có alias và đường dẫn cũ không đủ để xác định
  seed nên bị từ chối.

Tên thư mục thực tế trên Drive và hash checkpoint thật chưa thể xác nhận
offline; nếu không đúng mẫu hoặc entry đã mất thì cần người phụ trách cung
cấp đúng artifact, không nới locator để bỏ qua lỗi.

## Dữ liệu và tiền xử lý

PhoBERT dùng nguyên `text_seg` đã tách từ. ViSoBERT dùng `text` qua
`preprocess_text(..., emoji_mode="keep")`, giống nhau ở hai miền. Mỗi job
lọc `subset == test` trước khi xác thực/chuẩn hóa/tokenize văn bản. Đầu vào
train/dev không đi qua tokenizer hay predictor. Inference giữ FP32, không
autocast; batch sắp theo độ dài và hoàn trả đúng thứ tự ID gốc.

File dữ liệu đóng băng, manifest và `HARD_MERGE_RULE` được giữ nguyên.
VSMEC-strict lọc nhãn gốc 0..4 theo `docs/vsmec_strict_rule.md`;
692 − 37 − 129 = 526 câu. Macro-F1 strict chỉ lấy negative/positive,
nhưng dự đoán neutral vẫn sai và vẫn hiện trong ma trận 3×3/JS.

`scripts.check_step2_data.aggregate_probs_7to3` đã tồn tại và được gọi lại
qua wrapper kiểm tra xác suất ở `src/zs/mapping.py`. Wrapper dùng hằng số
chung; không khai báo thêm bảng ánh xạ độc lập.

### Hợp đồng data_hash

NEU giữ SHA-256 của `neuesc_100.csv` như Bước 1; đọc byte để xác minh nguồn
không sử dụng nhãn/văn bản train/dev cho suy luận. Với VSMEC, giữ công thức
SHA-256 của chuỗi `filename:sha256\n` cho train/dev/test theo đúng thứ tự
Bước 1. Digest train/dev lấy từ manifest đóng băng; inference không đọc
hai CSV đó. Digest của test được đối chiếu với byte thật trước suy luận.
Preflight F2 chịu trách nhiệm đối chiếu các artifact còn lại với manifest.
Không đổi ý nghĩa `data_hash` thành hash của một tập con test khác.

Chạy lại kiểm tra schema/ID/data_hash trước SKIP; smoke test phải dùng
thư mục kết quả riêng, tránh trộn cùng run ID với lượt đầy đủ.

## Tham chiếu và phân tích

Bốn job `zsA`, `zsB`, `refV`, `refN` xuất lần lượt 2/2/5/1 dòng mỗi
cặp alias/seed, tổng 100 dòng cho hai backbone × năm seed. Chỉ số được
tính từ CSV dự đoán đã lưu. Analyzer tiếp tục đọc dự đoán và xác minh
ID/nhãn giữa các seed/arm; không tin điểm số ghi tay trong metadata CSV.

Chỉ bước **báo cáo mốc sàn** đọc nhãn train của miền được chấm, không đọc
văn bản train/dev và không đưa prior vào predictor. Mốc ngẫu nhiên là
trung bình 2.000 mô phỏng có seed cố định. Với strict, mốc dùng prior
train VSMEC ánh xạ đầy đủ theo hợp đồng, nên có thể dự đoán neutral.

Mọi kiểm tra triển khai dùng dữ liệu giả/encoder nhỏ local. Kiểm tra
nhất quán Bước 1 nằm ở `src/zs/check_consistency.py`; schema Stage 1 cũ
không có dòng test được báo bỏ qua, không dùng điểm dev thay test.

## Việc người phụ trách cần thực hiện

Đưa đủ 20 checkpoint baseline và artifact hash/kết quả Bước 1 lên đúng
cấu trúc Drive; chạy notebook smoke trong thư mục riêng, rồi 40 lượt thật
và đối chiếu `refN`/`refV` với Bước 1. Các cổng F1/F3/G1/G2 còn thiếu
đầu vào không được tự coi là đã qua nhờ các kiểm thử giả lập.

## Kết quả xác minh local (14/11/2026)

- `.venv/Scripts/python.exe -X utf8 -m pytest -q`: **61 passed** trước
  thay đổi tương thích hash manifest khi chuyển checkpoint Colab → Drive.
- Sau thay đổi hẹp đó và hai regression mới, toàn bộ nhóm
  `tests/test_zs_infer.py`: **19 passed** (gồm chấp nhận manifest nằm trong
  đúng run và từ chối manifest alias toàn cục có đường dẫn cũ).
- CLI `src.zs.infer --help`, `src.zs.analyze --help` và biên dịch Python
  đều thành công. Bảng phân tích đã được kiểm tra trên 10 cặp giả
  backbone/seed, đủ 100 dòng và tái lập nguyên byte khi chạy lại.
- Readiness hiện tại: `E1=PASS`, `E2=SKIP`, `E3=PASS`;
  còn `F1, F3, G1, G2` thiếu đầu vào thật. Không chạy nghiên cứu thật.
- Mã đang ở workspace local, chưa stage/commit/push.
