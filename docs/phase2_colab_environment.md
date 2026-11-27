# Quy ước môi trường Colab cho notebook Giai đoạn 2

Cập nhật: **27/11/2026**. Profile tham chiếu: **`colab-task06-v1`**.
Áp dụng khi viết notebook Task07 và các notebook Phase2 tiếp theo.

Tài liệu này thống nhất cách chuẩn bị và kiểm tra môi trường, không thay đổi
giao thức `phase2-v1`, siêu tham số hay các cấu hình đóng băng. Một lượt smoke
thành công là bằng chứng chạy được pipeline DFT SA; không chứng nhận mọi
phương pháp, GPU hoặc phiên bản thư viện khác đều tương thích.

## 1. Nguồn bằng chứng

- [Notebook Task06 tải về từ Colab](../notebooks/phase2_task06_colab.ipynb):
  có cell cài `python3.13-venv` do người chạy bổ sung và output thực tế.
- [environment.json](../results/phase2/task06/2026-11-27_SA_DFT_f10_s0_A100/environment/environment.json).
- [pip-freeze.txt](../results/phase2/task06/2026-11-27_SA_DFT_f10_s0_A100/environment/pip-freeze.txt).
- [install.log](../results/phase2/task06/2026-11-27_SA_DFT_f10_s0_A100/logs/install.log)
  và [train.log](../results/phase2/task06/2026-11-27_SA_DFT_f10_s0_A100/logs/train.log).
- [Báo cáo smoke](../results/phase2/task06/2026-11-27_SA_DFT_f10_s0_A100/smoke_report.md).

Commit huấn luyện: `af12db20672b198accf724cbd26cdbf59b32d162`.
Task06 đã chạy train/dev, ngắt–resume, early stopping và SKIP; best dev
Macro-F1 là `0.6637240821785192`. Không dùng điểm này để chọn môi trường
hay thay cho sweep LR Task07.

## 2. Profile tham chiếu được chọn cho các notebook sau

Ưu tiên duy trì bộ phiên bản thực tế của Task06 cho các lượt Phase2 sắp tới.
Nếu Colab không cung cấp đúng Torch/Python này, dừng bước preflight, ghi nhận
khác biệt và kiểm thử môi trường thay thế trước khi chạy hàng loạt. Không
âm thầm coi phiên bản mới là cùng profile, không tự ép cài lại CUDA/Torch
chỉ để vượt assertion. Môi trường thay thế có tên/profile và output riêng.

| Thành phần | Giá trị thực tế Task06 | Quy ước |
|---|---|---|
| Python của tiến trình train | `3.13.18` | Đối chiếu đúng interpreter venv |
| PyTorch | `2.14.1+cu130` | Kế thừa bản GPU Colab, kiểm tra trước train |
| CUDA do Torch báo | `13.0` | Phân biệt với CUDA tối đa do driver hỗ trợ |
| GPU | `NVIDIA A100-SXM4-80GB` | Mốc đo chi phí đã có bằng chứng |
| Precision | `bf16`, strict | CUDA và BF16 phải được tiến trình train xác nhận |
| transformers | `4.57.1` | Pin |
| huggingface-hub | `0.36.2` | Pin |
| tokenizers | `0.22.2` | Pin |
| numpy | `2.1.3` | Pin |
| pandas | `3.1.0` | Pin |
| scikit-learn | `1.6.1` | Pin |
| scipy | `1.16.3` | Pin theo snapshot Task06 |
| PyYAML | `6.0.2` | Pin |
| sentencepiece | `0.2.2` | Pin |
| protobuf | `7.36.2` | Pin |
| safetensors | `0.8.0` | Pin theo snapshot Task06 |
| tqdm | `4.67.3` | Pin theo snapshot Task06 |

Các phiên bản trong bảng lấy từ artifact, không phải danh sách “mới nhất”.
`accelerate==1.15.0` có trong snapshot nhưng DFT/Seq hiện dùng trainer PyTorch
riêng; chỉ đưa vào bộ dependency bắt buộc khi notebook thực sự cần nó.
Task10 có LoRA/LLRD phải kiểm thử thêm dependency/hook của Task10.

`requirements.txt` hiện pin **Torch 2.9.0** và `requirements.lock.txt` là
snapshot khác. Không sửa hai file đó chỉ để khớp Colab, cũng không ghi rằng
Task06 đã dùng Torch 2.9.0. Khi so sánh baseline cũ, giữ provenance lịch sử
chưa xác minh cho tới khi có config/log thật.

## 3. Thứ tự chuẩn cho mọi notebook

1. Tham số: URL repo, nhánh/ref, tên task, output, nguồn checkpoint nếu cần.
2. Kiểm tra GPU CUDA/BF16 sơ bộ ở kernel.
3. Fetch repo theo nhánh, ghi commit đầy đủ, cố định checkout trong lượt chạy.
4. Kiểm tra hỗ trợ venv; cài đúng gói hệ thống nếu interpreter thiếu `venv`.
5. Tạo venv theo task/profile, cài dependency bằng **Python của venv**.
6. Preflight bằng Python của venv: phiên bản, CUDA/BF16, import pipeline,
   một phép forward/loss nhỏ và kiểm tra hữu hạn trên GPU trước sweep.
7. Kiểm tra dữ liệu/subset/protocol và checkpoint nguồn; lưu provenance.
8. Chạy CLI trong subprocess, log ra file và theo dõi tiến độ.
9. Kiểm tra artifact/hash, chế độ dev/test, metric và completion.
10. Xuất môi trường, log, báo cáo, ZIP; ghi rõ có/không kèm resume/checkpoint.

Không gọi train nếu cell cài đặt/preflight thất bại. Không có cell mặc định
train khi chỉ mở notebook hoặc xem `--help`.

## 4. Venv: bài học từ notebook đã chạy

Notebook tải về đã cài thành công `python3.13-venv` bằng apt trước khi tạo
venv. Đây là bước chuẩn bị đã có bằng chứng; output không đủ để kết luận
mọi runtime Colab đều thiếu gói này. Notebook mới phải xử lý việc thiếu
`venv`/`ensurepip` có thông báo rõ thay vì tiếp tục với venv tạo dở.

Không chép nguyên cell `rm -rf /content/task06-venv` sang mọi notebook.
Chọn thư mục mới, ví dụ `/content/phase2-task07-colab-task06-v1-venv`, khi
cần làm sạch; không xóa môi trường đang có tiến trình chạy. Với Python khác
3.13, không tiếp tục cài gói `python3.13-venv` rồi cho rằng interpreter đã đúng.

Khung tạo và cài package cho notebook mới (chạy trên Colab, không phải Windows):

```python
from pathlib import Path
import subprocess
import sys

VENV_DIR = Path('/content/phase2-task07-colab-task06-v1-venv')
PYTHON = VENV_DIR / 'bin' / 'python'
if not PYTHON.is_file():
    subprocess.run([
        sys.executable, '-m', 'venv', '--system-site-packages', str(VENV_DIR)
    ], check=True)

PINS = [
    'transformers==4.57.1', 'huggingface-hub==0.36.2', 'tokenizers==0.22.2',
    'numpy==2.1.3', 'pandas==3.1.0', 'scikit-learn==1.6.1', 'scipy==1.16.3',
    'PyYAML==6.0.2', 'sentencepiece==0.2.2', 'protobuf==7.36.2',
    'safetensors==0.8.0', 'tqdm==4.67.3',
]
# Notebook phải lưu stdout/stderr của bước cài vào install.log.
subprocess.run([str(PYTHON), '-m', 'pip', 'install', *PINS], check=True)
```

Đây là mẫu cho notebook sau, chưa phải cell đã được thực thi với toàn bộ
bộ pin tường minh trên một runtime Colab mới. Task06 chỉ cài trực tiếp tập
pin nhỏ hơn và kế thừa các thư viện còn lại; bảng trên ghi phiên bản thực tế.

`--system-site-packages` giúp dùng Torch GPU sẵn có nhưng **không cách ly hoàn
toàn** khỏi package Colab. Dùng `PYTHON -m pip`, `PYTHON -m src.phase2.cli`,
`PYTHON -m src.phase2.select_lr`; không dùng `!pip` hay import ở kernel để
suy ra phiên bản thực tế của subprocess train. Không replay toàn bộ hơn
700 package trong `pip-freeze.txt`, vốn có package Colab/path cục bộ không
phải dependency của dự án.

## 5. Preflight bắt buộc ở đúng tiến trình train

Ví dụ kiểm tra phần môi trường (notebook còn phải kiểm tra data/model):

```python
check = r'''
import platform
import importlib.metadata as metadata
import torch
from src.phase2.run import run_phase2_experiment

assert platform.python_version() == '3.13.18', 'Python profile mismatch'
assert torch.__version__ == '2.14.1+cu130', 'Torch profile mismatch'
assert torch.version.cuda == '13.0', 'CUDA build mismatch'
assert torch.cuda.is_available(), 'CUDA unavailable in training interpreter'
assert torch.cuda.is_bf16_supported(), 'Strict BF16 unsupported'
expected = {
    'transformers': '4.57.1', 'huggingface-hub': '0.36.2',
    'tokenizers': '0.22.2', 'numpy': '2.1.3', 'pandas': '3.1.0',
    'scikit-learn': '1.6.1', 'scipy': '1.16.3', 'PyYAML': '6.0.2',
    'sentencepiece': '0.2.2', 'protobuf': '7.36.2',
    'safetensors': '0.8.0', 'tqdm': '4.67.3',
}
for name, version in expected.items():
    assert metadata.version(name) == version, f'{name} profile mismatch'
print('GPU:', torch.cuda.get_device_name(0))
print('Profile preflight: PASS')
'''
subprocess.run([str(PYTHON), '-u', '-c', check], cwd=REPO_DIR, check=True)
```

Không tự thay BF16 strict bằng FP32 khi không đạt. GPU khác A100 có thể
được dùng sau preflight/kiểm thử riêng, nhưng không coi chi phí đo trên GPU
đó là cùng mốc A100. Không mặc định tên “G4” là một loại GPU cụ thể; ghi tên
do Torch và `nvidia-smi` trả về.

## 6. Cảnh báo đã gặp và cách xử lý

| Hiện tượng | Bằng chứng Task06 / xử lý cho notebook sau |
|---|---|
| `Checkout has changes` ngay sau clone | Bản đầu dùng `--no-checkout`, khiến index/worktree trống bị coi là sửa. Clone bình thường; chỉ phục hồi clone trống khi ngoài `.git` không có file và index rỗng. Không `reset --hard` checkout có sửa thật. |
| Venv không tạo được | Kiểm tra gói `pythonX.Y-venv` theo interpreter; log apt có cài `python3.13-venv`. Không gọi đường dẫn Python venv chưa tồn tại. |
| Pip báo không gỡ được package “outside environment” | Có trong install.log vì dùng system-site-packages. Không gỡ package hệ thống để chữa; kiểm tra phiên bản được import trong venv. |
| Pip báo conflict pandas/protobuf/HF Hub | Có trong log với Google Colab, cuDF, ydf, gradio, diffusers. Task06 pipeline vẫn chạy được, nhưng môi trường không sạch dependency toàn cục. Lưu `pip check`/install.log; không tự nâng pin hoặc import thêm các package xung đột vào subprocess train. Nếu pipeline cần chúng, tạo và xác minh profile mới. |
| Pandas cảnh báo Bottleneck `1.4.2` cũ | Đã thấy trong log, chưa gây lỗi Task06. Không ghi rằng đã sửa. Muốn nâng Bottleneck phải kiểm thử riêng và ghi phiên bản mới. |
| TensorFlow báo không thấy CUDA driver | Xuất hiện cùng log PyTorch chạy `device=cuda`, BF16 trên A100. Xác nhận GPU qua Torch của tiến trình train; không sửa CUDA chỉ dựa vào dòng TensorFlow. Chưa cần TensorFlow cho pipeline này. |
| ViSoBERT báo pooler khởi tạo mới | Code `Stage1Classifier` dùng `last_hidden_state[:, 0, :]` và head riêng. Pooler không được dùng để tạo logits. Kiểm tra đúng kiến trúc; không thay classifier chỉ để hết warning. |
| Log đứng lâu, I/O rất lớn | Trainer cũ lưu toàn bộ resume mỗi update. Bản `af12db2` lưu ở update đầu và mỗi lần eval, log mỗi 10 update và khi eval/lưu. Giữ cơ chế này; không đưa code cũ trở lại. |

Không hứa “không bao giờ lỗi”: Colab thay image, driver và package nền. Quy
ước này giúp phát hiện khác biệt trước train và giữ log đủ để sửa đúng lỗi.

## 7. Repo, resume và output

- Mặc định fetch nhánh `phase2-resource-transfer`; ghi SHA đầy đủ rồi cố
  định commit cho toàn bộ sweep. Không fetch code mới giữa 12 lượt Task07.
- Repo riêng tư dùng secret `GITHUB_TOKEN`, không ghi token vào URL/log/ZIP.
- Giữ config/data/subset/manifest đóng băng. Preflight xác minh ID/hash,
  không tạo lại subset để vượt lỗi. Sweep/smoke chỉ chuẩn bị train/dev.
- Resume phải cùng code, data, settings, nguồn checkpoint và môi trường.
  Khi nhánh tiến thêm commit, dùng SHA cũ để phục hồi lượt cũ. Không sửa
  fingerprint/completion để ép dùng resume không khớp.
- Dùng `/content` cho train/resume đang chạy. Drive là lựa chọn lưu trữ;
  ghi file resume lớn trực tiếp lên Drive có thể nghẽn I/O. Nếu sao lưu,
  chỉ sao lưu checkpoint đã hoàn tất ghi nguyên tử và xác minh hash bản sao.
- Output tách theo task, commit và profile. Có thể thêm session ID để tránh
  đè `environment.json` của lượt cũ khi runtime thay môi trường.
- Giữ tên run ID và các file trong completion; đổi tên thư mục bao ngoài
  được nhưng phải giữ quan hệ đường dẫn artifact. Không chuẩn hóa LF/CRLF
  của file gốc sau export. Trên Windows đối chiếu source với Git blob/LF.
- ZIP mặc định không có `resume.pt`: đọc lại kết quả được nhưng không đủ để
  resume hoặc làm checkpoint C0. Muốn tiếp tục phải xuất cả trạng thái cần thiết.
- Lưu notebook đã chạy làm bằng chứng; notebook mẫu phát hành sau này nên
  có output trống. Không sửa notebook bằng chứng để khiến nó giống mẫu mới.

## 8. Artifact môi trường phải xuất cho mỗi phiên

`environment.json` cần có profile ID, Git SHA, Python, `sys.executable`,
Torch/CUDA/cuDNN, GPU/VRAM/driver, BF16 support, phiên bản package thực tế,
`module.__file__` của Torch/Transformers/NumPy/Pandas và data/source hashes.
Task06 chưa ghi đủ mọi trường này; đây là yêu cầu cho notebook mới.

Kèm `install.log`, `pip-freeze.txt`, kết quả `pip check`, lệnh chạy không
chứa secret, log train, config hiệu dụng, provenance, metrics/cost,
completion và kết quả kiểm tra artifact. Ghi thời gian setup/download riêng
với train_seconds; ghi rõ peak VRAM của lần fit cuối khi có ngắt–resume.

## 9. Điều kiện trước khi chạy Task07

- Profile và GPU preflight đạt; đã ghi nhận các conflict kế thừa.
- Có checkpoint nguồn ViSoBERT của cả hai miền, seed 0, được locator kiểm
  tra; không dùng checkpoint smoke Task06 thay cho nguồn Seq.
- Dùng cùng môi trường và commit cho DFT/Seq, hai hướng và toàn bộ LR.
- Đủ 12 lượt dev-only ở fraction 10%, seed 0; không có prediction test.
- Gọi `src.phase2.select_lr` khi đủ lưới; hòa chọn LR nhỏ hơn; xuất cả
  `selection_p1_lr.json` và `p1_lr_selection.csv`.
- Nếu phải đổi profile giữa sweep, giữ output cũ riêng và chạy lại nhóm
  sweep đồng nhất trước khi khóa LR. Không chọn môi trường bằng điểm test.

File này là quy ước cho việc viết notebook tiếp theo. Chưa sửa notebook
Task06 tải về, chưa thay requirements/config, chưa chạy GPU mới, và không
coi mọi đề xuất kiểm tra ở trên là đã được thử trên Colab.
