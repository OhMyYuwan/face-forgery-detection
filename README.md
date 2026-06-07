# Face & Audio Forgery Detection

<div align="center">
  <img src="assets/github-header-banner.png" alt="SuperLeaf Banner" width="100%">
</div>


Image and audio forgery detection system based on natural trace extraction and filtering.

## HuggingFace Resources

- Image Model Weights: [OhMyYuwan/face-forgery-detection](https://huggingface.co/OhMyYuwan/face-forgery-detection)
- Audio Model Weights: [OhMyYuwan/audio-forgery-detection](https://huggingface.co/OhMyYuwan/audio-forgery-detection)
- Space Demo: [OhMyYuwan/face-forgery-detection](https://huggingface.co/spaces/OhMyYuwan/face-forgery-detection)

## Project Structure

```
face-forgery-detection/
├── datasets/
│   ├── image/                     # Image datasets
│   └── audio/                     # Audio datasets
│       └── asvspoof2019_la/
├── OhMyYuwan/
│   ├── face-forgery-detection/    # Image model registry and weights
│   └── audio-forgery-detection/   # Audio model registry and weights
├── scripts/
│   ├── image/                     # Image evaluation and generic three-stage training scripts
│   │   ├── evaluation/
│   │   └── train/
│   └── audio/                     # Audio evaluation and training scripts
│       ├── evaluation/
│       └── train/
├── Space/                         # Gradio demo application
├── results/
├── evaluate_parallel.sh
├── space_manager.sh
└── README.md
```

## Image Models

| Model | Backbone | Input Size |
|-------|----------|------------|
| convnext_base | ConvNeXt-Base | 224x224 |
| dinov2_base | DINOv2-Base | 224x224 |
| fastervit_2 | FasterViT-2 | 224x224 |
| inceptionnext_base | InceptionNeXt-Base | 224x224 |
| internvit_300m | InternViT-300M | 448x448 |
| mambavision_t | MambaVision-T | 224x224 |
| maxvit_base | MaxViT-Base | 224x224 |

## Audio Models

| Model | Backbone | EER |
|-------|----------|-----|
| wav2vec2_base | facebook/wav2vec2-base | 0.10% |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/OhMyYuwan/face-forgery-detection
cd face-forgery-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` requires CUDA and may need to be built from source on some systems:
> ```bash
> pip install mamba-ssm --no-build-isolation
> ```

### 3. Download Model Files

```bash
git lfs install
git clone https://huggingface.co/OhMyYuwan/face-forgery-detection OhMyYuwan/face-forgery-detection
```

Or skip large files initially and pull later:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/OhMyYuwan/face-forgery-detection OhMyYuwan/face-forgery-detection
cd OhMyYuwan/face-forgery-detection && git lfs pull && cd ../..
```

### Launch Gradio Demo

```bash
./space_manager.sh start    # Start service
./space_manager.sh status   # Check service status
./space_manager.sh stop     # Stop service
```

### Model Inference

```python
import json
import importlib.util
from pathlib import Path
import torch

model_name = "convnext_base"
model_dir = Path("OhMyYuwan/face-forgery-detection") / model_name

with open(model_dir / "config.json", "r") as f:
    config = json.load(f)

spec = importlib.util.spec_from_file_location(f"{model_name}.model", model_dir / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.OurNet(config)
state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
state_dict = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# Inference
with torch.no_grad():
    _, det = model.forward_det(image_tensor)
    score = torch.sigmoid(det).item()
```

## Image Training Scripts

All image training scripts are located in [scripts/image/train/](scripts/image/train/). The training entry point is a **generic three-stage template**: representation learning, feature selection, and forgery detection.

### Script Overview

| Script | 功能 | 用法 | 输出 |
|--------|------|------|------|
| [scripts/image/train/train_face_forgery.py](scripts/image/train/train_face_forgery.py) | 三阶段训练主脚本：Stage 1 表征学习 + Stage 2 特征筛选 + Stage 3 伪造检测 | `python scripts/image/train/train_face_forgery.py --gpu 0 --dataset_root <path> --deepfake_methods all --backbone <your_model> --savepath <savepath>` | `<savepath>/stage1_models_<backbone>/` + `<savepath>/stage2_core_select/` + `<savepath>/stage3_detnet_enhance/` |
| [scripts/image/train/train_natural_trace.py](scripts/image/train/train_natural_trace.py) | 训练基础组件（`OurNet`、`SupConLoss`、数据增强、LR 调度等），被主脚本导入 | 不直接运行 | - |
| [scripts/image/train/run.sh](scripts/image/train/run.sh) | 三阶段训练启动脚本，顶部使用 `your_*` 占位配置 | `bash scripts/image/train/run.sh` | `log/<backbone>_core_select_gpu<id>.log` + 三阶段模型权重 |

### Three-Stage Training Pipeline

**Stage 1 — 表征学习（Representation Learning）**

仅使用真实图像学习 real 共享的自然痕迹表征。核心损失包括：

- `SupConLoss`：异构特征对比损失。
- `MSELoss`：同质特征与 batch 内锚点的一致性约束。
- `CosineEmbeddingLoss`：异构与同质特征的正交性约束。

**Stage 2 — 特征筛选（Feature Selection）**

加载 Stage 1 权重后，将 `aux_fc2` 初始化到 `det_fc1`，在不改变模型结构的前提下微调 `det_fc1`。目标是让 homo 分支仍然能描述 real 共享特征，同时降低 fake 中也存在的、已被学习到的自然痕迹对后续检测的干扰。

核心损失项：

- `selection_contrast_weight * SupConLoss`：维持 real/fake 条件下的判别结构。
- `selection_fake_proto_weight * fake_proto_loss`：惩罚 fake 与 real-shared homo 原型过度对齐。
- `selection_real_compact_weight` / `selection_real_view_weight`：弱约束 real 特征仍保持合理紧致和增强一致性。
- `selection_retain_weight`：保留 real 强于 fake 的特征维度倾向。

默认只微调 `det_fc1`。如果需要在 Stage 2 同步极小幅度微调 backbone，可在 `run.sh` 中设置 `STAGE2_UPDATE_BACKBONE=true`。

**Stage 3 — 伪造检测（Forgery Detection）**

解冻 backbone、`det_fc1` 和 `det_fc2`，冻结辅助头 `aux_fc1/aux_fc2`。Stage 3 同时使用：

- `SupConLoss`：约束 `det_fc1` 输出的 trace/homo 特征。
- `BCEWithLogitsLoss`：训练 `det_fc2` 的真假二分类输出。

评估时使用每个 method 的 `split='test'` 数据，并用 per-method 平均指标作为 `[Eval Overall/Test]` 和 `best_model_acc.pth` 的保存依据。

### Dataset Layout

训练脚本期望以下目录结构（对应 `--dataset_root`）：

```text
<dataset_root>/
├── <method_1>/
│   ├── 0_real/
│   └── 1_fake/
├── <method_2>/
│   ├── 0_real/
│   └── 1_fake/
└── ...
```

`--deepfake_methods` 接受 `all`（自动发现）或逗号分隔的子集（如 `progan,stylegan2`）。每个 method 的 real/fake 图像分别按文件名排序后取前 90% 作训练、后 10% 作测试。

### Training Workflow

**1. 修改 `scripts/image/train/run.sh` 顶部的用户配置区域**

```bash
GPU_IDS=("your_gpu_id")
DATASET_ROOT="your_dataset_root"
DEEPFAKE_METHODS="your_deepfake_methods"  # use "all" or comma-separated methods
SAVE_PATH="your_save_path"
BACKBONE="your_model"
STAGE1_MODEL=""                           # set to a checkpoint to reuse Stage 1
```

如果选择 FasterViT、MambaVision 或旧的 InceptionNext checkpoint 路径，需要同步修改 [scripts/image/train/train_natural_trace.py](scripts/image/train/train_natural_trace.py) 中的 `your_pretrained_weights/...` 占位路径。

**2. 启动三阶段训练**

```bash
bash scripts/image/train/run.sh
```

`run.sh` 会检查 `your_*` 占位是否已经替换；如果未替换，会直接报错退出，避免误训练。

**3. 监控训练**

```bash
tail -f scripts/image/train/log/<backbone>_core_select_gpu*.log
watch -n 1 nvidia-smi
```

### Key Hyperparameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage1_epochs` | 50 | Stage 1 表征学习轮数 |
| `--stage1_lr` / `--stage1_head_lr` | `1e-5` / `0.01` | 主干 / 投影头学习率 |
| `--stage2_epochs` | 20 | Stage 2 特征筛选轮数 |
| `--stage2_lr` | `1e-4` | `det_fc1` 特征筛选学习率 |
| `--stage2_lr_backbone` | `1e-6` | 启用 `--stage2_update_backbone` 时的 backbone 学习率 |
| `--selection_fake_proto_weight` | `1.5` | fake-to-real 原型排斥损失权重 |
| `--selection_retain_weight` | `0.05` | real 强于 fake 特征维度的保留权重 |
| `--stage3_epochs` | 50 CLI / 100 run.sh | Stage 3 伪造检测轮数 |
| `--stage3_lr` | `0.01` CLI / `0.001` run.sh | `det_fc2` 分类头学习率 |
| `--stage3_lr_backbone` | `1e-5` | Stage 3 backbone 和 `det_fc1` 微调学习率 |
| `--lambda_aux` | `0.3` | Stage 3 辅助对比损失权重 |
| `--temperature` | `0.1` | SupCon 温度 |
| `--img_size` | 224 | 输入分辨率 |
| `--eval_interval` | 20 CLI / 10 run.sh | Stage 3 每 N 个 epoch 在 test split 上评估一次 |

### Output Files

```text
<savepath>/gpu<id>/
├── stage1_models_<backbone>/
│   ├── best_model.pth
│   ├── ckpt_epoch_50.pth
│   └── final_epoch.pth
├── stage1_tensorboard_<backbone>/
├── stage2_core_select/
│   ├── path_models_<backbone>/
│   │   ├── best_model.pth
│   │   ├── feature_score.npy
│   │   ├── feature_weight_score.npy
│   │   ├── feature_real_mean.npy
│   │   └── feature_fake_mean.npy
│   └── path_tensorboard_<backbone>/
└── stage3_detnet_enhance/
    ├── path_models_<backbone>/
    │   ├── best_model.pth
    │   ├── best_model_acc.pth
    │   ├── ckpt_epoch_20.pth
    │   └── final_epoch.pth
    └── path_tensorboard_<backbone>/
```

## Evaluation Scripts

All evaluation scripts are located in `scripts/image/evaluation/`.

| Script | 功能 | 用法 | 输出 |
|--------|------|------|------|
| `evaluate_parallel.sh` | 并行评估所有模型（自动分配多GPU） | `bash evaluate_parallel.sh` | `results/evaluate/logs/*.log` |
| `scripts/image/evaluation/evaluate.py` | 评估单个模型 | `python scripts/image/evaluation/evaluate.py --model internvit_300m --device cuda` | `results/evaluate/{model}/metrics.json` |
| `scripts/image/evaluation/evaluate_ensemble.py` | 评估模型组合性能（加权平均 / 投票法） | `python scripts/image/evaluation/evaluate_ensemble.py --models m1 m2 --method voting` | `results/evaluate/ensemble/*.json` |
| `scripts/image/evaluation/optimize_thresholds.py` | 为每个模型搜索最优检测阈值 | `python scripts/image/evaluation/optimize_thresholds.py` | `OhMyYuwan/face-forgery-detection/optimal_thresholds.json` |
| `scripts/image/evaluation/view_results.py` | 快速查看所有模型的评估结果 | `python scripts/image/evaluation/view_results.py` | 终端表格输出 |

### Evaluation Workflow

**1. 评估所有模型（并行）**
```bash
bash evaluate_parallel.sh
```

**2. 优化检测阈值**
```bash
python scripts/image/evaluation/optimize_thresholds.py
```

**3. 评估模型组合**
```bash
# 投票法
python scripts/image/evaluation/evaluate_ensemble.py --models convnext_base fastervit_2 inceptionnext_base --method voting

# 加权平均
python scripts/image/evaluation/evaluate_ensemble.py --models convnext_base fastervit_2 inceptionnext_base --method weighted
```

**4. 查看结果**
```bash
python scripts/image/evaluation/view_results.py
```
