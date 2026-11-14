# AGENT_zscdt_core.md: code Bước 2 (zero-shot VSMEC ↔ NEU-ESC) kèm bootstrap CI

Dành cho AI agent làm việc trên repo `PhuongNgoVan/Xdomain-Transfer-Learning`, nhánh **`zscdt-core`** (người phụ trách tạo sẵn từ `zero-shot-cross-domain-transfer`; agent không tự tạo, đổi tên hay merge nhánh). Đọc hết file này trước khi viết code.

## 0. Nhiệm vụ và ranh giới

Viết code **suy luận** và **phân tích** cho Bước 2 của Workflow v3.0 (mục 6 của kế hoạch v3.0), gồm:

1. Chạy checkpoint Bước 1 trên test của miền kia (và trên test của chính miền để làm tham chiếu): không train, không fine-tune.
2. Tính macro-F1 và các chỉ số liên quan theo ba "góc nhìn nhãn", transfer gap, retention, bất đối xứng, JS divergence, ma trận nhầm lẫn, mốc sàn.
3. Gắn **khoảng tin cậy 95% bằng bootstrap hai tầng** (lấy lại mẫu câu test và seed) cho mọi chỉ số chính.

Việc **ngoài phạm vi nhánh này** (để các nhánh `zscdt-*` sau): oracle prior correction, phân tích theo nhóm câu, đo độ lệch nội tại (PAD, t-SNE/UMAP, Jaccard/KL từ vựng). Chỉ làm độ nhạy ánh xạ ở mức đã nêu ở mục 4.

**Quy tắc cứng**

1. **Không dùng dữ liệu đích, kể cả dev và train**, ở bất kỳ bước suy luận nào. Với mỗi lượt, chỉ **test** của miền được chấm mới được tokenize. Không chỉnh ngưỡng, không hiệu chỉnh tiên nghiệm bằng dữ liệu đích.
2. Không sửa `data/clean/*`, `MANIFEST.json`, `HARD_MERGE_RULE`, `src/plm/stage1.py`, `src/plm/train_neu3.py` hay các test cũ. Chỉ thêm file mới (và thêm hằng số, không đổi hằng số cũ).
3. Mọi con số đưa vào báo cáo phải do hàm trong `src/` tính và có test; notebook chỉ gọi hàm, vẽ hình, xuất bảng.
4. Thứ tự nhãn 3: 0 negative, 1 neutral, 2 positive. Thứ tự 7: Enjoyment 0, Sadness 1, Fear 2, Anger 3, Disgust 4, Surprise 5, Other 6. Quy tắc gộp lấy từ `HARD_MERGE_RULE` / `HARD_MERGE_GROUPS`; không viết lại quy tắc một lần nữa trong code mới.
5. Suy luận ở **FP32**. Seed bootstrap cố định để kết quả lặp lại.
6. Agent **không truy cập Drive**: viết và test bằng dữ liệu giả; không chạy các lượt thật. Việc chạy thật làm bằng notebook Colab (người phụ trách).

## 1. Đọc nhánh trước khi code

Ghi kết quả vào `docs/zs_notes.md`:

- [ ] `Stage1Classifier.load_stage1_checkpoint` nạp được **cả** checkpoint 7 lớp (VSMEC) và 3 lớp (NEU-3), của cả PhoBERT và ViSoBERT; trả về mô hình + tokenizer. Nếu không, dừng và báo.
- [ ] Tiền xử lý đầu vào: PhoBERT dùng cột `text_seg`; ViSoBERT dùng cột `text` với `emoji_mode: keep` áp **giống nhau** cho cả hai miền (xem `preprocess_text` và cách `stage1.py` / `train_neu3.py` gọi). Suy luận phải dùng đúng tiền xử lý đã dùng lúc train checkpoint đó.
- [ ] Tên thư mục checkpoint thật: VSMEC theo mẫu `{alias}_Stage1-(?:base-)?(?:emokeep-)?bf16_s{seed}_lr{lr}` (ví dụ `PhoBERT_Stage1-bf16_s0_lr2e-5`, không bắt buộc chữ `base`); NEU-3 `{alias}_NEU3-base[-emokeep]-bf16_s{seed}_lr{lr}`; mỗi thư mục chứa `encoder/` và `head.pt` (kiểm bằng code, không đoán).
- [ ] `data/clean/vsmec_{train,dev,test}.csv` và `neuesc_100.csv` (`subset ∈ {train, dev, test}`); cột nhãn 7 của VSMEC; cách tính `data_hash` đã dùng ở Bước 1 (dùng lại đúng hàm đó).
- [ ] Hàm cộng xác suất 7→3 đã có trong repo chưa (nếu có thì dùng lại).

## 2. Dữ liệu, góc nhìn nhãn (label view) và tham chiếu

| label_view | Tập chấm | Nhãn thật | Ghi chú |
| --- | --- | --- | --- |
| `V3-NEU` | `neuesc_100.csv`, `subset == test` | 3 nhãn gốc của NEU | |
| `V3-VSMEC` | `vsmec_test.csv` | nhãn 7 gộp 3 bằng `HARD_MERGE_RULE` | bản đầy đủ |
| `V3-VSMEC-strict` | tập con của `vsmec_test` | như trên | **chỉ giữ câu có nhãn gốc Enjoyment, Sadness, Fear, Anger, Disgust (bỏ Surprise, Other)** (kế hoạch v3.0 mục 3). Nhãn thật chỉ có negative và positive; macro-F1 tính trung bình trên **hai lớp {neg, pos}**; mô hình dự đoán neutral vẫn tính là sai. Ghi cờ `in_strict` ở từng câu trong file dự đoán. |
| `V7-VSMEC` | `vsmec_test.csv` | nhãn 7 gốc | chỉ cho tham chiếu VSMEC (kiểm cổng "F1 khớp số công bố") |

Bốn loại lượt (**job**) cho mỗi (backbone, seed), 5 seed × 2 backbone × 4 = 40 lượt:

| job | Checkpoint | Chấm trên | Vai trò |
| --- | --- | --- | --- |
| `zsA` | VSMEC (7 lớp) | test NEU | zero-shot chiều A |
| `zsB` | NEU-3 (3 lớp) | test VSMEC | zero-shot chiều B |
| `refV` | VSMEC (7 lớp) | test VSMEC | tham chiếu trong miền cho chiều B |
| `refN` | NEU-3 (3 lớp) | test NEU | tham chiếu trong miền cho chiều A |

Gộp 7 → 3 cho đầu 7 lớp, hai cách (`map_method`):

- `sum` (chính): cộng xác suất softmax theo `HARD_MERGE_GROUPS`, rồi lấy argmax trên 3 nhóm;
- `argmax`: argmax 7 lớp rồi ánh xạ bằng `HARD_MERGE_RULE`.

Đầu 3 lớp: `map_method = none`.

Cặp so sánh (cùng tập test, cùng không gian nhãn):

- Chiều A, `V3-NEU`: ZS = `zsA` (sum), REF = `refN`.
- Chiều B, `V3-VSMEC`: ZS = `zsB`, REF = `refV` ánh xạ `sum`.
- Chiều B, `V3-VSMEC-strict`: ZS = `zsB`, REF = `refV` ánh xạ `sum` trên cùng tập strict.
- Độ nhạy ánh xạ: thêm bản `argmax` cho mọi chỗ có đầu 7 lớp (`zsA`, `refV`).

## 3. File cần tạo

| File | Nội dung |
| --- | --- |
| `configs/zs.yaml` | `backbones` (alias, tên HF, `text_col`, `emoji_mode`), `seeds: [0,1,2,3,4]`, `batch_size: 128`, `boot: 10000`, `boot_seed: 0`, `ci: 0.95` |
| `src/zs/__init__.py` | |
| `src/zs/ckpt.py` | `locate_checkpoint(root, alias, domain, seed)` (domain ∈ `vsmec`, `neu3`; bỏ thư mục có `sweep`/`smoke` trong tên; phải đúng **một** kết quả; kiểm tra **cả** `encoder/` có `config.json` + `model.safetensors` hoặc `pytorch_model.bin`, **và** `head.pt` có `weight.shape[0]` = 7 cho VSMEC hoặc 3 cho NEU-3, không nạp encoder; nếu có file hash thì đối chiếu **cả hash encoder lẫn head**, không chấp nhận manifest rỗng/thiếu entry cho checkpoint được chọn — chặn R08) |
| `src/zs/views.py` | dựng các tập chấm ở mục 2, hàm `strict_mask`, hàm gộp nhãn (dùng `HARD_MERGE_RULE`) |
| `src/zs/mapping.py` | gộp `sum` và `argmax`; hàm thuần, vector hóa |
| `src/zs/infer.py` | CLI mục 5 và `run_job(...)`, `build_zs_run_id`, `JOB_RUN_TAG` |
| `src/stats/__init__.py`, `src/stats/bootstrap.py` | bootstrap hai tầng mục 6 (tái dùng được cho Bước 3) |
| `src/zs/analysis.py` | gap, retention, bất đối xứng, JS, ma trận, mốc sàn; gọi `bootstrap.py` |
| `src/zs/analyze.py` | CLI mục 7, ghi các bảng ở mục 7 |
| `src/common/schema.py` | **thêm** `ZS_COLUMNS`, `ZS_PRED_COLUMNS` (không đổi schema cũ) |
| `tests/test_zs_*.py` | các test ở mục 9 |
| `docs/zs_metric_definitions.md` | định nghĩa chỉ số (chép từ mục 4 và 6 của file này) |
| `docs/zs_notes.md` | kết quả mục 1, mọi chỗ khác giả định, câu hỏi cho người phụ trách |

## 4. Chỉ số

Với mỗi lượt và mỗi label view: macro-F1, weighted-F1, accuracy, F1 từng lớp. Gọi `compute_metrics` hiện có nếu khớp, nếu không viết hàm riêng và **kiểm bằng `sklearn`**.

- Nhãn dùng để tính macro-F1: `V3-*` đầy đủ: {0,1,2}; `V3-VSMEC-strict`: {0,2}; `V7-VSMEC`: {0..6}. Lớp không có dự đoán và không có mẫu thật cho F1 bằng 0 (`zero_division=0`, như Bước 1).
- **Mốc sàn (1.3)** cho từng view: majority-class và ngẫu nhiên theo tiên nghiệm, dùng tiên nghiệm của **split train của chính miền được chấm** (V3-NEU: train NEU; hai view VSMEC: nhãn train VSMEC gộp 3). Ngẫu nhiên: trung bình macro-F1 của 2.000 lần mô phỏng, seed cố định. Đây là mốc báo cáo, không đi vào bất kỳ mô hình nào.
- **Transfer gap** = F1_REF − F1_ZS (đơn vị điểm macro-F1). **Retention** = F1_ZS / F1_REF. Trong đó F1_* là **trung bình qua seed** của macro-F1 trên cùng view. Cặp REF/ZS theo mục 2.
- **Bất đối xứng:** (a) giữa hai chiều: retention_A − retention_B (view B chính là `V3-VSMEC`, thêm `V3-VSMEC-strict`), theo từng backbone; (b) giữa hai backbone: retention_PhoBERT − retention_ViSoBERT, theo từng chiều. Thêm cả hiệu của gap.
- **JS divergence** (log cơ số 2, nằm trong [0, 1]) giữa phân bố nhãn dự đoán và phân bố nhãn thật trên cùng view; tính theo từng seed. Thêm JS giữa tiên nghiệm nhãn train của hai miền (V3-VSMEC train vs NEU train) một lần.
- **Ma trận:** (i) 3×3 (thật × dự đoán, ba nhãn) cho mỗi view và mỗi (backbone, role); (ii) **7×3** cho chiều B: hàng là cảm xúc gốc của test VSMEC, cột là nhãn 3 mà checkpoint NEU dự đoán; (iii) **3×7** cho chiều A: hàng là nhãn thật NEU, cột là cảm xúc mà checkpoint VSMEC dự đoán (argmax 7). Cộng đếm qua 5 seed rồi chuẩn hóa hàng (xuất cả đếm lẫn chuẩn hóa).

## 5. CLI suy luận

```
python -m src.zs.infer --config configs/zs.yaml --backbone PhoBERT --seed 0 \
  --job zsA|zsB|refV|refN \
  --vsmec-ckpt-root <dir> --neu-ckpt-root <dir> \
  --data-dir data/clean --results-dir <dir> [--max-examples N] [--batch-size 128]
```

Notebook phụ thuộc vào hợp đồng này; đổi tên gì phải báo.

**Export** từ `src/zs/infer.py`: `build_zs_run_id(alias, job, seed) -> str` với dạng `{alias}_{TAG}-fp32_s{seed}`, `TAG` = `ZSV2N` (zsA), `ZSN2V` (zsB), `REFV2V` (refV), `REFN2N` (refN), ví dụ `PhoBERT_ZSV2N-fp32_s0`; và `run_job(...)`, `main()`.

**Hành vi**

1. Tìm checkpoint bằng `locate_checkpoint`; nạp; in dòng `[Precision] FP32` và đường dẫn checkpoint.
2. Chỉ tokenize **test** (xem quy tắc 1); `--max-examples N` lấy N câu đầu của test (cho smoke test).
3. Dự đoán FP32, batch sắp theo độ dài, lưu thứ tự `id` gốc.
4. Ghi `{results_dir}/predictions_zs/{run_id}_{eval_set}.csv` (`eval_set` là `neuesc_test` hoặc `vsmec_test`) với cột `ZS_PRED_COLUMNS`: `id, y_true3, y_true7, y_pred3_sum, y_pred3_argmax, y_pred7, p3_neg, p3_neu, p3_pos, p7_0..p7_6, in_strict`. Với đầu 3 lớp: `y_pred3_sum = y_pred3_argmax` là argmax 3 lớp, `y_pred7 = -1`, các cột `p7_*` để trống (NaN). Với tập NEU: `y_true7 = -1`, `in_strict` trống.
5. Tính chỉ số **từ chính file dự đoán đã ghi**, rồi thêm dòng vào `{results_dir}/zs.csv`.
6. `=== [DONE] {run_id}: n=..., macro_f1=..., elapsed=...s ===` khi xong; `=== [SKIP] run_id=... ===` nếu mọi dòng cần có của `run_id` đã có trong `zs.csv`. Lỗi: mã thoát khác 0 và in `Traceback`.
7. Ghi dòng vào `zs.csv` theo khóa `(run_id, label_view, map_method)` (không trùng).
8. Thiết kế để test được: `run_job(..., predictor=None)` nhận một predictor giả (trả về xác suất ngẫu nhiên có seed) để test chạy trên CPU không cần mô hình.

**`zs.csv`** (`ZS_COLUMNS`): `run_id, backbone, stage1, stage2, eval_domain, label_view, mode, fraction, seed, split, map_method, n_eval, macro_f1, weighted_f1, accuracy, f1_per_class, f1_neg, f1_neu, f1_pos, ckpt_path, src_domain, tgt_domain, precision, code_version, data_hash, config_path`.

Giá trị: `stage1` = miền train của checkpoint (`VSMEC` hoặc `NEU`), `stage2 = ""`, `eval_domain` = `NEU` hoặc `VSMEC`, `mode` = `zs` khi `stage1 != eval_domain`, ngược lại `ref`; `fraction = 100`; `split = "test"`; `f1_per_class` là chuỗi JSON (độ dài 3, hoặc 7 với `V7-VSMEC`); `f1_neg/neu/pos` để trống ở view không có lớp đó. `precision = "fp32"`.

**Số dòng mỗi (backbone, seed)** (tổng 100 dòng cho 10 cặp):

| job | dòng |
| --- | --- |
| `zsA` | `V3-NEU` × {sum, argmax} |
| `zsB` | `V3-VSMEC`, `V3-VSMEC-strict` (map `none`) |
| `refV` | `V7-VSMEC` (none); `V3-VSMEC` × {sum, argmax}; `V3-VSMEC-strict` × {sum, argmax} |
| `refN` | `V3-NEU` (none) |

## 6. Bootstrap hai tầng (`src/stats/bootstrap.py`)

Mục đích: khoảng tin cậy 95% cho macro-F1, F1 từng lớp, JS, gap, retention và các hiệu bất đối xứng. Bước 2 **không kiểm định giả thuyết, không p-value, không Holm** (đó là Bước 3); chỉ báo cáo ước lượng và khoảng tin cậy.

**Đơn vị lấy lại mẫu:** (1) **câu test** (n câu, có hoàn lại) và (2) **seed** (S = 5, có hoàn lại). Một lần lặp: rút trọng số câu `w` (nhiều thức đa thức, tương đương lấy lại chỉ số có hoàn lại) và một bộ seed; thống kê của một "nhánh" (arm) là **trung bình qua các seed đã rút** của chỉ số tính trên tập câu đã lấy lại (cùng `w` cho mọi seed trong arm).

**Ghép cặp**

- Các arm **cùng tập test** (ví dụ ZS và REF ở chiều A; hai backbone ở cùng chiều) dùng **chung `w`** trong từng lần lặp; seed được rút **độc lập cho từng arm** (vì seed s của hai arm là hai mô hình khác nhau, không có cặp thật).
- Các arm **khác tập test** (chiều A so với chiều B) dùng `w` **độc lập**.
- gap, retention, hiệu: tính **trong từng lần lặp** từ thống kê các arm rồi lấy phân vị 2,5 và 97,5. Retention chia cho giá trị tham chiếu gần 0 (< 1e-8) thì đánh dấu NaN và đếm số lần.

**Cài đặt**

- Mặc định `B = 10000`, `numpy.random.Generator(PCG64(boot_seed))`; kết quả lặp lại được.
- Tính theo khối (chunk 500 lần lặp) để không cần lưu ma trận B×n; vector hóa: với `w` (chunk×n) và các ma trận one-hot của nhãn thật/dự đoán (n×K), TP, số dự đoán, số thật của mỗi lớp là các tích ma trận; F1 từng lớp rồi trung bình đúng nhãn của view (mục 4).
- Giá trị điểm (point estimate) tính trên toàn bộ dữ liệu, **không** lấy trung bình bootstrap. Ghi thêm SE bootstrap và độ lệch chuẩn đơn giản giữa các seed (không qua bootstrap).
- API tối thiểu: `bootstrap_arm(y_true, preds_by_seed, labels, stat, ...)`, `bootstrap_contrast(arms, fn, share_test)`, trả về `{point, ci_lo, ci_hi, se, n_nan}`. Cấu trúc đủ tổng quát để Bước 3 gọi lại cho "sequential − baseline".
- Ghi cảnh báo trong `docs/zs_metric_definitions.md`: với 5 seed, tầng lấy lại seed ước lượng phương sai giữa seed thô; luôn báo thêm std giữa seed. Test VSMEC (692 câu, strict 526 câu) nhỏ nên khoảng tin cậy rộng, không kết luận từ chênh lệch vài điểm.

## 7. CLI phân tích và bảng đầu ra

```
python -m src.zs.analyze --results-dir <dir> --data-dir data/clean \
  --out-dir <dir>/tables [--boot 10000] [--seed 0] [--allow-missing]
```

Đọc `zs.csv` và `predictions_zs/`. Mặc định lỗi nếu có nhóm (backbone, job) nào không đủ 5 seed (`--allow-missing` để chạy một phần, in rõ nhóm thiếu). Kết thúc in `=== [ANALYZE DONE] ... ===`. Chạy hai lần với cùng tham số phải ra bảng **y hệt**.

Bảng ghi vào `tables/` (notebook đọc đúng các tên và cột này):

| File | Hàng | Cột |
| --- | --- | --- |
| `zs_main.csv` | (backbone, direction ∈ {A,B}, label_view) | `backbone, direction, target, label_view, n_eval, n_seeds, zs_f1, zs_std, zs_ci_lo, zs_ci_hi, ref_f1, ref_std, ref_ci_lo, ref_ci_hi, gap, gap_ci_lo, gap_ci_hi, retention, ret_ci_lo, ret_ci_hi, floor_majority, floor_prior_random` |
| `zs_asymmetry.csv` | (kind, backbone hoặc direction, label_view, stat) | `kind` ∈ {`direction`, `backbone`}, `contrast`, `label_view`, `stat` ∈ {`retention`, `gap`}, `diff, ci_lo, ci_hi` |
| `zs_js.csv` | (backbone, direction, label_view, role ∈ {zs, ref}) | `js_pred_true_bits, std, ci_lo, ci_hi`; thêm một dòng `kind = prior` cho JS tiên nghiệm hai miền |
| `zs_per_class.csv` | (backbone, direction, label_view, role, class ∈ {neg, neu, pos}); riêng `V3-VSMEC-strict` **bỏ qua lớp `neu`, chỉ xuất `neg` và `pos`** | `f1, std, ci_lo, ci_hi` |
| `zs_mapping_sensitivity.csv` | (backbone, label_view, role, map_method ∈ {sum, argmax}) | `f1, std, ci_lo, ci_hi` |
| `zs_floors.csv` | (label_view, floor ∈ {majority, prior_random}) | `macro_f1, n_sim` |
| `zs_confusion3_{backbone}_{direction}_{label_view}_{role}_{counts\|rownorm}.csv` | 3×3 | nhãn hàng, cột neg/neu/pos |
| `zs_matrix_7x3_{backbone}_{counts\|rownorm}.csv` | 7×3 (chiều B) | hàng theo tên 7 cảm xúc |
| `zs_matrix_3x7_{backbone}_{counts\|rownorm}.csv` | 3×7 (chiều A) | cột theo tên 7 cảm xúc |
| `zs_meta.json` | | `boot, boot_seed, ci, code_version, data_hash, n_rows, định nghĩa chỉ số (phiên bản)` |

`direction` A = VSMEC → NEU (`target = NEU`), B = NEU → VSMEC (`target = VSMEC`).

## 8. Kiểm tra nhất quán với Bước 1 (script riêng `src/zs/check_consistency.py`)

Hàm nhận `--baseline-csv` (NEU: `neu3.csv`; VSMEC: `stage1.csv`) và so với kết quả tái tính của Bước 2:

- `refN` macro-F1 test từng seed (tính lại từ checkpoint) khớp dòng `split == test` của dòng baseline cùng seed trong `neu3.csv` trong khoảng 1e-3; in bảng chênh lệch; WARN nếu lệch lớn (nghĩa là checkpoint, tiền xử lý hoặc dữ liệu không còn khớp Bước 1).
- `refV` (V7) so với `stage1.csv` nếu file có dòng test; nếu không có thì ghi "không có dòng test, bỏ qua" (không coi là lỗi). Đây cũng là chỗ kiểm cổng 1.4 (so với số đã công bố): chỉ in số, ngưỡng do người phụ trách đặt.
- Khi có `--baseline-pred-dir` (thư mục `predictions3` của Bước 1.2): tỉ lệ `y_pred` trùng nhau giữa file Bước 1 và `refN` tái tính, mong đợi gần 100%.

## 9. Test bắt buộc (CPU, dữ liệu giả, không cần mạng hay Drive)

- **`mapping`/`views`:** `sum` khác `argmax` ở ví dụ xác suất cho trước (ví dụ: Enjoyment 0,30 và các nhãn tiêu cực 0,15/0,10/0,15/0,10 → `sum` ra negative, `argmax` ra positive); tổng hàng bằng 1; thứ tự cột `(neg, neu, pos)`. Tập strict đúng bằng câu có nhãn gốc {0..4}; với dữ liệu thật (nếu có) strict n = 692 − 37 − 129 = 526 (bỏ qua nếu thiếu dữ liệu).
- **`ckpt`:** `locate_checkpoint` trên thư mục giả: đúng một kết quả; bỏ qua `sweep`/`smoke`; lỗi khi không có hoặc có hai; kiểm `weight.shape[0]`.
- **Quy tắc vàng 1:** chạy `run_job` với tokenizer/predictor giả ghi lại mọi chuỗi đầu vào; tất cả chuỗi phải thuộc test của miền được chấm, và không có chuỗi nào thuộc train/dev của miền đích. Test này phải **fail** nếu ai đó cho dev đích vào.
- **`run_job`:** dòng và file đầu ra đúng như mục 5 (đủ cột, đúng số dòng mỗi job, `p3` cộng bằng 1, `y_pred3_sum` khớp `argmax(p3)`); chạy lại ra `[SKIP]` và không thêm dòng.
- **Bootstrap:** (i) bằng **sklearn** khi dùng cùng chỉ số lấy lại: sinh chỉ số có hoàn lại, tính `f1_score(..., average="macro", labels=...)` trên mẫu lấy lại và so với bản vector hóa bằng trọng số `bincount` (sai khác < 1e-12); (ii) lặp lại với cùng seed ra y hệt, khác seed ra khác; (iii) bộ phân loại hoàn hảo, S = 1: khoảng tin cậy suy biến về điểm; (iv) hai arm cùng tập test dùng chung `w` (kiểm qua hàm nội bộ trả `w`); (v) mô phỏng độ phủ: với phân loại có độ chính xác thật đã biết, tỉ lệ khoảng tin cậy chứa giá trị thật nằm trong 90 đến 99% (đánh dấu `slow`, B nhỏ); (vi) view strict dùng nhãn {0,2}; (vii) retention với tham chiếu ≈ 0 trả NaN và đếm.
- **`analysis`:** gap, retention, bất đối xứng ra đúng trên bảng giả có đáp án tay; JS bằng 0 khi hai phân bố trùng, bằng 1 khi tách rời (log2); ma trận 7×3 có tổng bằng số câu × số seed; mốc sàn majority đúng với tay.
- **`analyze`:** chạy hai lần ra bảng giống hệt; thiếu seed thì lỗi, có `--allow-missing` thì chạy và in nhóm thiếu; mọi tên và cột bảng đúng mục 7.
- Các test cũ (`tests/test_stage1.py` và còn lại) vẫn xanh.

## 10. Quy trình chạy trên Colab (để agent hiểu cách mã được dùng)

Notebook `zscdt_step2_colab.ipynb`: mount Drive, `git clone` nhánh `zscdt-core`, kiểm tra đủ 4 nhóm × 5 seed checkpoint, smoke test (`--max-examples 32`), chạy 40 lượt `src.zs.infer`, đối chiếu với Bước 1, chạy `src.zs.analyze`, rồi chỉ **đọc** các bảng để vẽ hình (rừng khoảng tin cậy, heatmap 7×3, ma trận nhầm lẫn, phân bố nhãn). Thư mục checkpoint (người phụ trách đã dùng ở Bước 1): `Xdomain/finetune-vsmec-7c/{phobert,visobert}/ckpt_baseline` và `Xdomain/finetune-neu-3c/{phobert,visobert}/ckpt_baseline`. Kết quả Bước 2 lưu ở `Xdomain/zscdt-step2/`.

## 11. Cổng nghiệm thu

Agent chỉ được báo "xong" khi:

1. `pytest` xanh, gồm test cũ.
2. `run_job` với predictor giả chạy hết 4 job cho một (backbone, seed) và `analyze` chạy hết trên 10 cặp giả, ra đủ bảng mục 7.
3. `python -m py_compile` sạch trên mọi file mới.
4. `docs/zs_notes.md`, `docs/zs_metric_definitions.md` đầy đủ.

## 12. Giao nộp

- Commit nhỏ, thông điệp rõ, trên nhánh `zscdt-core`. Không force-push, không đụng `data/`.
- Báo cáo cuối ngắn: file đã tạo, kết quả `pytest`, các điểm trong `docs/zs_notes.md` cần người phụ trách quyết, và mọi chỗ khác với hợp đồng ở mục 5 và 7.
- Không chạy lượt thật. Không tự đặt ngưỡng cổng; chỉ in số.
