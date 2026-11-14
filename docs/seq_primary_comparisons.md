# Bốn so sánh chính seq-v1

Commit cùng config trước lượt nghiên cứu thật đầu tiên. Macro-F1 test,
alpha0.05, paired bootstrap hai tầng B10000; Holm m=4. Baseline là REFN2N
hoặc REFV2V cùng seed/test/view. Gain=seq−base. Thiếu primary thì chưa
kết luận significant; mọi so sánh khác là thăm dò.

```json
[
  {"id":"P1","backbone":"PhoBERT","direction":"SA","label_view":"V3-NEU","stat":"macro_f1"},
  {"id":"P2","backbone":"ViSoBERT","direction":"SA","label_view":"V3-NEU","stat":"macro_f1"},
  {"id":"P3","backbone":"PhoBERT","direction":"SB","label_view":"V7-VSMEC","stat":"macro_f1"},
  {"id":"P4","backbone":"ViSoBERT","direction":"SB","label_view":"V7-VSMEC","stat":"macro_f1"}
]
```

Giới hạn và định nghĩa p_boot: `seq_metric_definitions.md`. Trạng thái
commit và config/log baseline thật phải được kiểm tra trước tuyên bố xác nhận.
