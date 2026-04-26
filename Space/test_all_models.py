#!/usr/bin/env python3
"""
测试所有模型的推理功能

用法：
    python test_all_models.py <image_path>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Apply compatibility patches
import compat_patches

import torch
import json
import importlib.util
from torchvision import transforms
from PIL import Image
import time

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    """加载单个模型"""
    model_dir = f"../face-forgery-detection/{model_name}"
    config_path = f"{model_dir}/config.json"
    model_py_path = f"{model_dir}/model.py"
    weight_path = f"{model_dir}/pytorch_model.bin"

    with open(config_path) as f:
        config = json.load(f)

    spec = importlib.util.spec_from_file_location(f"{model_name}.model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.OurNet(config)
    state = torch.load(weight_path, map_location=_device)
    if isinstance(state, dict):
        sd = state.get("state_dict", state.get("model", state))
    else:
        sd = state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(_device)
    model.eval()
    return model, config


def test_model(model_name, model, config, image):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    try:
        # Get input size from config
        input_size = config.get("input_size", [224, 224])
        if isinstance(input_size, list):
            input_size = tuple(input_size)

        # Create transform for this model
        model_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = model_transform(image).unsqueeze(0).to(_device)

        start_time = time.time()

        with torch.no_grad():
            _, det_head = model.forward_det(image_tensor)
            score = torch.sigmoid(det_head).item()

        elapsed = time.time() - start_time

        label = "不安全 (伪造)" if score >= 0.5 else "安全 (真实)"
        print(f"✅ 推理成功")
        print(f"   输入尺寸: {input_size}")
        print(f"   伪造分数: {score:.4f}")
        print(f"   判断结果: {label}")
        print(f"   推理耗时: {elapsed:.3f}秒")

        return True, score

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    if len(sys.argv) < 2:
        print("用法: python test_all_models.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        sys.exit(1)

    print(f"🖼️  加载图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"   图像尺寸: {image.size}")
    print(f"   设备: {_device}")

    # 读取模型注册表
    registry_path = "../face-forgery-detection/OhMyYuwan/face-forgery-detection/registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    model_names = list(registry["models"].keys())
    print(f"\n📋 发现 {len(model_names)} 个模型")

    results = {}

    for model_name in model_names:
        try:
            print(f"\n⏳ 加载模型: {model_name}...")
            model, config = load_model(model_name)
            print(f"✅ 模型加载成功")

            success, score = test_model(model_name, model, config, image)
            results[model_name] = {"success": success, "score": score}

            # 释放内存
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            results[model_name] = {"success": False, "score": None}

    # 汇总结果
    print(f"\n{'='*60}")
    print("📊 测试汇总")
    print(f"{'='*60}")

    successful = [name for name, res in results.items() if res["success"]]
    failed = [name for name, res in results.items() if not res["success"]]

    print(f"\n✅ 成功: {len(successful)}/{len(model_names)}")
    if successful:
        print("\n模型名称                    | 伪造分数 | 判断")
        print("-" * 60)
        for name in successful:
            score = results[name]["score"]
            label = "不安全" if score >= 0.5 else "安全"
            print(f"{name:25} | {score:8.4f} | {label}")

    if failed:
        print(f"\n❌ 失败: {len(failed)}")
        for name in failed:
            print(f"   - {name}")

    # 多模型投票
    if len(successful) > 1:
        scores = [results[name]["score"] for name in successful]
        avg_score = sum(scores) / len(scores)
        final_label = "不安全 (伪造)" if avg_score >= 0.5 else "安全 (真实)"

        print(f"\n🗳️  多模型投票结果:")
        print(f"   平均分数: {avg_score:.4f}")
        print(f"   最终判断: {final_label}")


if __name__ == "__main__":
    main()
