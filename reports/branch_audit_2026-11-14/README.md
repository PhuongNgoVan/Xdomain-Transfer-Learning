# Rà soát toàn bộ nhánh — 14/11/2026

Đã đối chiếu danh sách nhánh trực tiếp với GitHub bằng `git ls-remote --heads origin`. Sáu nhánh remote đều khớp commit local. Không checkout, merge, sửa mã nguồn hay sửa dữ liệu. Các file mới trong thư mục này chỉ phục vụ rà soát.

| Nhánh | Commit | Phát hiện liên quan |
| --- | --- | --- |
| `main` | `b4c5bbb` | R03, R05, R06, R07, R12 |
| `p1` | `85c3627` | R08, R09, R10; thiếu đầu vào Bước 2 |
| `phobert-vsmec-7c` | `f3e07c2` | R02, R03, R07 |
| `visobert-vsmec-7c` | `845d6b7` | R02, R03, R07 |
| `Neu3-phobert` | `79a70f6` | R01–R07, R11, R12 |
| `Neu3-visobert` | `763440a` | R01–R07, R11, R12 |

Nhánh local `zscdt-core` trùng commit với `p1`, nên có cùng kết quả. R02 và R04 trên hai nhánh NEU-3 áp dụng cho luồng `train_neu3.py`; R05–R06 áp dụng cho luồng Stage 1 còn dùng vòng lặp thủ công.

Mức P1: cần xử lý trước khi chạy/tin kết quả thí nghiệm. Mức P2: lỗi tính đúng đắn, kiểm tra hoặc phục hồi cần sửa trong đợt tiếp theo. Đây là rà soát mã và kiểm tra có mục tiêu, không phải chứng nhận rằng mọi lỗi đã được tìm thấy.

## Các lỗi cần xử lý

### R01 — P1 — Hai nhánh NEU-3 không import được entry point

`src/plm/train_neu3.py` tự thực hiện `from src.plm.train_neu3 import (...)`, yêu cầu `NUM_LABELS3` và bảy helper. File hiện tại chỉ định nghĩa `_make_compute_metrics`, `run_neu3_hf`, `main`; các helper cũ đã bị thay thế. Chạy module hoặc import vào test sẽ dừng với `ImportError`. Đồng thời `tests/test_neu3.py` vẫn import `run_neu3`, trong khi file chỉ còn `run_neu3_hf`.

Bằng chứng: [PhoBERT, dòng 54](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/79a70f609ba5461810f6058073d423fca7625f98/src/plm/train_neu3.py#L54), [ViSoBERT, dòng 59](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/763440a0685e14c02cc0cd5eaefca90550d952cf/src/plm/train_neu3.py#L59). Probe thực thi nguyên câu import lấy từ AST của từng nhánh đã tái hiện lỗi `cannot import name 'NUM_LABELS3'`.

Hướng sửa: khôi phục helper vào module dùng chung, import từ đó; thống nhất entry point và API mà test sử dụng. Đổi tên import đơn thuần không đủ vì phần định nghĩa đã mất.

### R02 — P1 — Dependency không đủ cho các luồng Trainer

Cả bốn nhánh model mới đều sử dụng `transformers.Trainer` nhưng `requirements.txt` và `requirements.lock.txt` không khai báo `accelerate`. Riêng `phobert-vsmec-7c` còn import `datasets.Dataset` ở hai module mà không khai báo `datasets`. Môi trường mới cài theo README/CI có thể dừng trước khi train.

Bằng chứng: [import datasets](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/f3e07c27cfa2daa15005ac45b7c02941d9a4f021/src/plm/tokenizer_dataset.py#L9), [requirements](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/f3e07c27cfa2daa15005ac45b7c02941d9a4f021/requirements.txt). Mã Transformers 4.57.1 đang có trong `.venv/Lib/site-packages/transformers/training_args.py:2224` cũng kiểm tra và yêu cầu Accelerate; metadata ghi đây là dependency của extra `torch`, không phải dependency bắt buộc của cài đặt `transformers` thường.

Hướng sửa: bổ sung phiên bản tương thích và cập nhật lock; xác minh cài đặt trong môi trường mới. Chưa thực hiện cài đặt hay kiểm thử môi trường mới trong lần rà soát này.

### R03 — P1 — Các seed ghi đè checkpoint của nhau khi dùng chung thư mục mặc định

Trên cả năm nhánh có mã train, `ckpt_subdir = c_dir / alias` không chứa seed hoặc run ID. Chạy seed 0 rồi seed 1 cùng backbone và cùng `--ckpt-dir` sẽ ghi đè `encoder/` và `head.pt`. Các CSV của lượt cũ vẫn còn nhưng checkpoint tương ứng đã mất. Stage 1 còn ghi hash theo alias, nên bản ghi hash cũng bị thay thế. Điều này cản trở yêu cầu giữ đủ checkpoint cho từng seed ở Bước 2.

Bằng chứng: [PhoBERT Stage 1, dòng 263](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/f3e07c27cfa2daa15005ac45b7c02941d9a4f021/src/plm/stage1.py#L263), [NEU-3, dòng 144](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/79a70f609ba5461810f6058073d423fca7625f98/src/plm/train_neu3.py#L144).

Hướng sửa: lưu theo run ID hoặc seed và ghi hash theo cùng khóa. Hiện có thể tránh ghi đè bằng cách truyền thư mục checkpoint riêng cho từng lượt; code không tự bảo đảm việc này.

### R04 — P1 — NEU-3 có lỗi chuẩn hóa loss khi gradient accumulation

Trên hai nhánh NEU-3, `Stage1Classifier.forward` nhận `**kwargs`, tính cross-entropy trung bình và bỏ qua `num_items_in_batch`. Class không đặt `accepts_loss_kwargs = False`. Với Transformers 4.57.1, Trainer vì vậy suy ra model tự xử lý số mẫu tích lũy và bỏ bước chia loss theo số micro-batch khi có `num_items_in_batch`. Cấu hình `gradient_accumulation_steps: 2` sẽ tích lũy gradient với thang đo sai, ảnh hưởng clipping và so sánh thí nghiệm.

Bằng chứng: [forward, dòng 52](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/79a70f609ba5461810f6058073d423fca7625f98/src/plm/model_ckpt.py#L52); đối chiếu mã thư viện local `transformers/trainer.py:633` và `:4059`. Hai nhánh VSMEC mới đã có cờ `accepts_loss_kwargs = False` nhưng thay đổi đó chưa có trong hai nhánh NEU-3.

Hướng sửa: khai báo đúng hợp đồng loss hoặc tính loss theo tổng mẫu tích lũy; kiểm tra gradient tương đương giữa batch đầy đủ và gradient accumulation. Đây là lỗi suy ra từ mã và nhánh điều kiện của thư viện, chưa chạy so sánh gradient. Luồng NEU-3 hiện còn bị R01 chặn trước đó.

### R05 — P1 — Resume Stage 1 có thể làm mất model tốt nhất

Trên `main` và hai nhánh NEU-3, checkpoint trung gian lưu trọng số hiện tại cùng `best_macro_f1` và `best_step`, nhưng không lưu trọng số best. Khi resume, code nạp trọng số hiện tại rồi gán chúng vào `best_state_dict`. Nếu best từng xuất hiện trước checkpoint cuối và các lần đánh giá tiếp theo không vượt best cũ, cuối train sẽ lưu trọng số sai trong khi metadata vẫn chỉ tới best cũ.

Bằng chứng: [main stage1.py, dòng 279](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/src/plm/stage1.py#L279), [trạng thái lưu, dòng 188](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/src/plm/trainer_utils.py#L188). Cùng logic tồn tại ở `Neu3-phobert:stage1.py:304` và `Neu3-visobert:stage1.py:332`.

Hướng sửa: lưu best state riêng hoặc lưu/nạp checkpoint được chỉ định bởi `best_step`; không suy best từ checkpoint mới nhất.

### R06 — P2 — Resume Stage 1 chạy lại phần epoch đã xử lý

Vẫn trên ba nhánh dùng vòng lặp thủ công, code lấy `start_epoch` từ checkpoint rồi dựng lại DataLoader từ đầu epoch; không lưu/khôi phục vị trí batch hoặc bỏ qua các batch đã chạy. Optimizer/scheduler vẫn được nạp ở step cũ. Sau gián đoạn giữa epoch, mẫu đầu epoch được học lại với lịch learning rate đã tiến thêm. Checkpoint cũng không lưu trạng thái RNG và GradScaler, nên chưa bảo đảm tái lập khi resume.

Bằng chứng: [main stage1.py, dòng 342](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/src/plm/stage1.py#L342), [các trường checkpoint](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/src/plm/trainer_utils.py#L188).

Hướng sửa: lưu vị trí xử lý, RNG và scaler; hoặc thống nhất dùng cơ chế resume của Trainer. Xác minh bằng một lượt bị ngắt giữa epoch so với lượt chạy liên tục.

### R07 — P2 — Lượt chạy thiếu artifact có thể bị bỏ qua vĩnh viễn

Các luồng Stage 1 kiểm tra có `run_id` trong `stage1.csv` là trả về `SKIP`. Nhưng CSV được ghi trước sidecar và hash checkpoint. Nếu phiên Colab/Kaggle ngắt ngay sau khi ghi CSV, lần chạy lại bỏ qua lượt chưa hoàn chỉnh, không tạo bù sidecar/hash. Mất sidecar còn khiến bộ đọc Bước 2 không khôi phục được điểm test từ schema Stage 1 cũ. Luồng NEU-3 có rủi ro tương tự sau khi ghi đủ hai dòng CSV nhưng chưa ghi sidecar/hash.

Bằng chứng: [điều kiện skip PhoBERT](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/f3e07c27cfa2daa15005ac45b7c02941d9a4f021/src/plm/stage1.py#L166), [ghi CSV trước sidecar/hash](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/f3e07c27cfa2daa15005ac45b7c02941d9a4f021/src/plm/stage1.py#L445).

Hướng sửa: chỉ đánh dấu hoàn tất sau khi đủ artifact; khi chạy lại phải xác minh và tạo bù các artifact còn thiếu.

### R08 — P1 — J3 có thể PASS dù encoder thay đổi hoặc hash manifest rỗng

Trên `p1`/`zscdt-core`, `check_group_j` chỉ so `head_sha256`. Trường `encoder_sha256` được bộ train ghi ra không được kiểm tra. Ngoài ra, chỉ cần một file hash JSON tồn tại, kể cả `{}`, J3 vẫn có thể PASS khi không bật reload. Không có bước bảo đảm hash bao phủ các checkpoint mà J1 đã tìm thấy.

Bằng chứng: [check_step2_data.py, dòng 2697](../../scripts/check_step2_data.py#L2697). Probe gọi nguyên hàm từ AST, dùng file tạm và thay bộ đọc head bằng stub trả shape, xác nhận cả hai trường hợp: sửa toàn bộ byte encoder vẫn `J3=PASS`; đổi JSON hash thành `{}` cũng `J3=PASS`.

Hướng sửa: đối chiếu encoder và head cho từng checkpoint đã tìm thấy, bắt buộc đủ entry, xác minh đường dẫn gắn đúng run/seed. Probe này kiểm chứng logic hash; không kiểm chứng tải tensor hay encoder thật.

### R09 — P2 — F1 từng lớp của dev bị gắn nhãn thành F1 test

Khi nạp schema `STAGE1_COLUMNS`, bộ đọc lấy test macro-F1 từ sidecar nhưng dựng dòng test bằng `{**base_rec, 'split': 'test', 'macro_f1': test_f1}`. `base_rec` chứa F1 từng lớp của dev. G3 sau đó tính trung bình F1 từng lớp trên những dòng mang nhãn test này, có thể đưa ra kết luận sai về lớp dưới ngưỡng 0,30.

Bằng chứng: [dòng 508](../../scripts/check_step2_data.py#L508), [G3 dùng dòng test](../../scripts/check_step2_data.py#L1987). Probe thực thi nguyên biểu thức dựng record xác nhận `f1_anger=0.9` của dev vẫn được giữ ở dòng test có `macro_f1=0.1`.

Hướng sửa: xuất/nạp F1 test thật từ prediction hoặc sidecar; nếu không có thì để thiếu và báo thiếu, không kế thừa từ dev.

### R10 — P2 — Reload-check có thể lấy prediction sai seed hoặc sai lượt chạy

`_find_prediction_csv_for_group` trả về CSV đầu tiên khớp alias và tên dataset, không ràng buộc seed/run ID của checkpoint đang kiểm tra. Khi nhiều seed/LR/variant nằm chung thư mục, J3 có thể so checkpoint đúng với logits của lượt khác rồi báo FAIL sai.

Bằng chứng: [dòng 2429](../../scripts/check_step2_data.py#L2429). Probe gọi nguyên hàm với checkpoint seed 4 và hai prediction seed 0/4 trong thư mục chung; hàm chọn file seed 0.

Hướng sửa: truyền run ID/seed được chọn vào bộ tìm file và yêu cầu khớp chính xác; báo thiếu hoặc mơ hồ nếu không xác định được.

### R11 — P2 — CSV NEU-3 làm hỏng lệnh gộp kết quả chung

Hai nhánh NEU-3 ghi `results/neu3.csv` theo `NEU3_COLUMNS`, nhưng `discover_result_files` chỉ loại `stage1.csv`, `token_stats.csv`, `merged_results.csv`. Nó thu cả `neu3.csv` rồi `merge_result_files` ép đọc theo `RESULT_COLUMNS`, dẫn đến `SchemaError` trong cách bố trí đầu ra mặc định.

Bằng chứng: [merge_results.py, dòng 19](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/79a70f609ba5461810f6058073d423fca7625f98/src/eval/merge_results.py#L19), [kiểm tra schema, dòng 120](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/79a70f609ba5461810f6058073d423fca7625f98/src/eval/merge_results.py#L120). Probe thực thi nguyên hàm discover xác nhận nó thu `neu3.csv`.

Hướng sửa: loại CSV chuyên biệt khỏi bộ gộp chung hoặc thêm bước chuyển đổi schema rõ ràng.

### R12 — P1 — Bộ test trên main và hai nhánh NEU-3 import module chưa có

`tests/test_build_wtrans.py` import `src.data.build_wtrans` và `src.data.polarity_kappa`; `tests/test_polarity_kappa.py` cũng import module thứ hai. Cả hai module đều không có trong Git tree của `main`, `Neu3-phobert`, `Neu3-visobert`. Đây là import trực tiếp, không có `importorskip`, nên pytest sẽ lỗi collection khi các dependency bên ngoài đã đủ. CI hiện gọi toàn bộ `pytest -v`.

Bằng chứng: [test_build_wtrans.py, dòng 32](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/tests/test_build_wtrans.py#L32), [test_polarity_kappa.py, dòng 28](https://github.com/PhuongNgoVan/Xdomain-Transfer-Learning/blob/b4c5bbb5d489be0f5ef8daf924301de3f72c37a4/tests/test_polarity_kappa.py#L28). Hai module triển khai đang có trên `p1`.

Hướng sửa: tích hợp triển khai tương ứng cùng test vào nhánh đích. Các module `heads`, `losses`, `callbacks`, `predict`, `train_plm` cũng chưa có nhưng test tương ứng chủ động dùng `importorskip`; không tính chúng thành lỗi collection đã xác nhận này.

## Việc còn thiếu và các hạn chế đã biết

- **Bước 2 chưa đủ đầu vào trên p1:** cấu hình vẫn để `vsmec_strict_file`, hai điểm chuẩn macro-F1 và bốn `ckpt_roots` là `null`; chưa có CSV Bước 1 ở thư mục inputs local. Báo cáo đã lưu ghi `CHƯA SẴN SÀNG: E1, E2, E3, F1, F3, G1, G2`. Đây là trạng thái thiếu đầu vào, không phải bằng chứng rằng các thí nghiệm trên Drive chưa từng được chạy.
- **Hai cặp gần trùng** đã được ghi nhận và xuất ID kiểm tra độ nhạy: VSMEC test `6311` ↔ NEU dev `25960`; NEU test `28482` ↔ VSMEC train `3607`. Các file dữ liệu chính giống hệt nhau trên mọi nhánh nên hạn chế dữ liệu này áp dụng chung. Không tự sửa tập test đóng băng.
- **Manifest polarity cũ:** trên mọi nhánh, hash Git blob của `polarity_long.csv` và `polarity_nhat.csv` lệch manifest clean. Trên p1 đây đã được ghi nhận là F2b/WARN ngoài phạm vi Bước 2. Byte của file local có thể khác Git blob do CRLF; cả hai cách đối chiếu ở workspace hiện tại vẫn cho thấy hai file polarity lệch manifest. Không tự viết lại manifest để làm kiểm tra PASS.
- **Môi trường local không chạy được pytest đầy đủ:** `.venv/pyvenv.cfg` trỏ tới Python 3.11 đã mất; `.venv/Scripts/python.exe --version` báo không tìm thấy Python gốc. Python trên PATH là MSYS2 3.12.7, chưa có pytest/numpy/pandas/torch/transformers. Chưa cài thêm hay thay đổi môi trường.
- **Tài liệu chưa thống nhất phiên bản quy trình:** quy ước cũ ghi seed 42–46, schema xác suất và VSFC; task/implementation Bước 2 mới dùng seed 0–4 và workflow v3.0. Cần chốt tài liệu chủ đạo khi tích hợp. Không kết luận seed 0–4 là lỗi vì task Bước 2 hiện yêu cầu rõ bộ seed này.

## Phạm vi xác minh và bằng chứng

- Đọc Git tree, diff và các luồng dữ liệu/train/evaluation/checkpoint của cả sáu nhánh; kiểm tra thêm nhánh local trùng commit.
- Phân tích AST toàn bộ file Python được track ở từng nhánh: không phát hiện lỗi cú pháp AST; phát hiện module/symbol thiếu và self-import. Các import chủ động skip được phân biệt ở R12.
- Đối chiếu SHA-256 từng entry manifest clean/splits bằng byte Git blob. Các manifest split đều khớp; các sai khác clean được ghi trong `evidence.json`.
- Năm file dữ liệu chính giống cùng Git blob trên tất cả nhánh. Số dòng: VSMEC train/dev/test = 5.542/686/692; NEU-ESC = 32.966; VSFC clean = 3.166. Không có ID trùng trong từng file được kiểm tra. Đây không thay thế kiểm tra trùng văn bản hay huấn luyện thực tế.
- Chạy thành công các probe chuẩn thư viện Python, lấy nguyên AST của code cần kiểm tra; có stub shape cho bộ đọc head trong probe J3. Không chạy full pytest, GPU, tải tokenizer/model hay truy cập checkpoint Drive.

File tái lập: [audit.py](audit.py), [evidence.json](evidence.json), [probes.py](probes.py), [probes.json](probes.json).

Lệnh đã chạy thành công từ thư mục repo:

```powershell
python reports/branch_audit_2026-11-14/audit.py
python reports/branch_audit_2026-11-14/probes.py
```

Ưu tiên xử lý: R01/R02/R12 để mở được luồng chạy và test; R03–R06 để bảo toàn checkpoint và tính đúng đắn của train; R08–R10 trước khi dựa vào kết luận readiness; sau đó R07/R11 và hoàn thiện đầu vào Bước 2.
