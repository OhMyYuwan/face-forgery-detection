# Face Forgery Detection

Image forgery detection system for the TIFS2026 project.

## HuggingFace Resources

- Model Weights: [OhMyYuwan/face-forgery-detection](https://huggingface.co/OhMyYuwan/face-forgery-detection)
- Space Demo: [OhMyYuwan/face-forgery-detection](https://huggingface.co/spaces/OhMyYuwan/face-forgery-detection)

## Project Structure

```
face-forgery-detection/
├── datasets/                  # Dataset
│   ├── train/
│   └── test/
├── OhMyYuwan/                 # Model registry and weights
│   └── face-forgery-detection/
│       ├── convnext_base/
│       │   ├── config.json
│       │   ├── model.py
│       │   └── pytorch_model.bin
│       ├── dinov2_base/
│       ├── fastervit_2/
│       ├── inceptionnext_base/
│       ├── internvit_300m/
│       ├── mambavision_t/
│       ├── maxvit_base/
│       ├── registry.json
│       └── optimal_thresholds.json
├── scripts/                   # Evaluation and training scripts
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── evaluate_ensemble.py
│   │   ├── optimize_thresholds.py
│   │   └── view_results.py
│   └── train/
├── Space/                     # Gradio demo application
│   ├── app.py
│   ├── compat_patches.py
│   └── test_all_models.py
├── results/
│   ├── evaluate/              # Evaluation outputs
│   └── train/                 # Training outputs
├── evaluate_parallel.sh       # Parallel evaluation script
├── space_manager.sh           # Gradio space management script
└── README.md
```

## Models

| Model | Backbone | Input Size |
|-------|----------|------------|
| convnext_base | ConvNeXt-Base | 224x224 |
| dinov2_base | DINOv2-Base | 224x224 |
| fastervit_2 | FasterViT-2 | 224x224 |
| inceptionnext_base | InceptionNeXt-Base | 224x224 |
| internvit_300m | InternViT-300M | 448x448 |
| mambavision_t | MambaVision-T | 224x224 |
| maxvit_base | MaxViT-Base | 224x224 |

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
./space_manager.sh start    # Start service in background
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

## Evaluation Scripts

All evaluation scripts are located in `scripts/evaluation/`.

| Script | 功能 | 用法 | 输出 |
|--------|------|------|------|
| `evaluate_parallel.sh` | 并行评估所有模型（自动分配多GPU） | `bash evaluate_parallel.sh` | `scripts/evaluation/results/{model}/metrics.json` + `results/evaluate/logs/*.log` |
| `scripts/evaluation/evaluate.py` | 评估单个模型 | `python scripts/evaluation/evaluate.py --model internvit_300m --device cuda` | `scripts/evaluation/results/{model}/metrics.json` |
| `scripts/evaluation/evaluate_ensemble.py` | 评估模型组合性能（加权平均 / 投票法） | `python scripts/evaluation/evaluate_ensemble.py --models m1 m2 --method voting` | `scripts/evaluation/results/ensemble/*.json` |
| `scripts/evaluation/optimize_thresholds.py` | 为每个模型搜索最优检测阈值 | `python scripts/evaluation/optimize_thresholds.py` | `OhMyYuwan/face-forgery-detection/optimal_thresholds.json` |
| `scripts/evaluation/view_results.py` | 快速查看所有模型的评估结果 | `python scripts/evaluation/view_results.py` | 终端表格输出 |

### Evaluation Workflow

**1. 评估所有模型（并行，推荐）**
```bash
bash evaluate_parallel.sh
```

**2. 优化检测阈值**
```bash
python scripts/evaluation/optimize_thresholds.py
```

**3. 评估模型组合**
```bash
# 投票法
python scripts/evaluation/evaluate_ensemble.py --models convnext_base fastervit_2 inceptionnext_base --method voting

# 加权平均
python scripts/evaluation/evaluate_ensemble.py --models convnext_base fastervit_2 inceptionnext_base --method weighted
```

**4. 查看结果**
```bash
python scripts/evaluation/view_results.py
```
