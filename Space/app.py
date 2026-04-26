"""
TIFS2026 — Image Forgery Detection Platform

Supports loading models from:
1. Local face-forgery-detection repository
2. HuggingFace Hub (OhMyYuwan/face-forgery-detection)
"""

import sys
import os
import json
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Apply compatibility patches BEFORE importing other libraries
import compat_patches

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Model Registry & Loader
# ---------------------------------------------------------------------------

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_models = {}
_model_configs = {}  # Store model configs for input size

# Try local first, fallback to HF Hub
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "OhMyYuwan/face-forgery-detection")
HF_REPO_ID = "OhMyYuwan/face-forgery-detection"
USE_LOCAL = os.path.exists(LOCAL_MODELS_DIR)


def load_model_from_local(model_name: str):
    """Load model from local face-forgery-detection directory"""
    model_dir = os.path.join(LOCAL_MODELS_DIR, model_name)
    config_path = os.path.join(model_dir, "config.json")
    model_py_path = os.path.join(model_dir, "model.py")
    weight_path = os.path.join(model_dir, "pytorch_model.bin")

    with open(config_path) as f:
        config = json.load(f)

    # Dynamic import model.py
    spec = importlib.util.spec_from_file_location(f"{model_name}.model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.OurNet(config)
    state = torch.load(weight_path, map_location=_device, weights_only=False)
    if isinstance(state, dict):
        sd = state.get("state_dict", state.get("model", state))
    else:
        sd = state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(_device)
    model.eval()
    return model, config


def load_model_from_hub(model_name: str):
    """Load model from HuggingFace Hub"""
    config_path = hf_hub_download(HF_REPO_ID, f"{model_name}/config.json")
    model_py_path = hf_hub_download(HF_REPO_ID, f"{model_name}/model.py")
    weight_path = hf_hub_download(HF_REPO_ID, f"{model_name}/pytorch_model.bin")

    with open(config_path) as f:
        config = json.load(f)

    spec = importlib.util.spec_from_file_location(f"{model_name}.model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.OurNet(config)
    state = torch.load(weight_path, map_location=_device, weights_only=False)
    if isinstance(state, dict):
        sd = state.get("state_dict", state.get("model", state))
    else:
        sd = state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(_device)
    model.eval()
    return model, config


def get_available_models():
    """Scan available models"""
    if USE_LOCAL:
        registry_path = os.path.join(LOCAL_MODELS_DIR, "registry.json")
        with open(registry_path) as f:
            registry = json.load(f)
        return list(registry["models"].keys())
    else:
        # Fallback: hardcoded list (or fetch from HF API)
        return ["convnext_base"]


AVAILABLE_MODELS = get_available_models()
DEFAULT_MODEL = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None

# Load optimal thresholds for each model
_optimal_thresholds = {}
_thresholds_path = os.path.join(LOCAL_MODELS_DIR, "optimal_thresholds.json")
if os.path.exists(_thresholds_path):
    with open(_thresholds_path) as f:
        _optimal_thresholds = {k: v['threshold'] for k, v in json.load(f).items()}
    print(f"✅ 已加载最优阈值配置: {_thresholds_path}")


def get_model_threshold(model_name: str) -> float:
    """获取模型的最优阈值"""
    return _optimal_thresholds.get(model_name, 0.5)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model(model_name: str) -> str:
    """Load a single model into global cache"""
    global _models, _model_configs
    if model_name in _models:
        return f"✅ 模型 {model_name} 已缓存"

    try:
        if USE_LOCAL:
            model, config = load_model_from_local(model_name)
        else:
            model, config = load_model_from_hub(model_name)
        _models[model_name] = model
        _model_configs[model_name] = config
        return f"✅ 模型 {model_name} 已加载 (设备: {_device})"
    except Exception as e:
        return f"❌ 加载失败: {e}"


# Pre-load default model
_startup_msg = _load_model(DEFAULT_MODEL) if DEFAULT_MODEL else "⚠️ 无可用模型"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(image: Image.Image, selected_models: list[str], method: str,
            global_threshold: float, enable_global: bool,
            model_weights: dict, model_thresholds: dict,
            locked_models: list[str]):
    """Multi-model inference with weighted average or voting"""
    if image is None:
        return "—", "请上传图像"

    if not selected_models:
        return "错误", "请至少选择一个模型"

    # Load models if not cached
    for model_name in selected_models:
        if model_name not in _models:
            msg = _load_model(model_name)
            if "❌" in msg:
                return "错误", msg

    scores = {}

    with torch.no_grad():
        for model_name in selected_models:
            model = _models[model_name]
            config = _model_configs[model_name]

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

            tensor = model_transform(image.convert("RGB")).unsqueeze(0).to(_device)
            _, det_head = model.forward_det(tensor)
            score = torch.sigmoid(det_head).item()
            scores[model_name] = score

    if not scores:
        return "错误", "没有可用的模型"

    # 应用阈值逻辑：
    # 1. 如果启用了全局阈值：未锁定的模型使用全局阈值，锁定的模型使用独立阈值
    # 2. 如果未启用全局阈值：所有模型使用各自的最优阈值（锁定的模型使用用户设置的独立阈值）
    thresholds = {}
    for model_name in selected_models:
        if model_name in locked_models:
            # 锁定的模型始终使用用户设置的独立阈值
            thresholds[model_name] = model_thresholds.get(model_name, 0.5)
        else:
            if enable_global:
                # 启用全局阈值：未锁定的模型使用全局阈值
                thresholds[model_name] = global_threshold
            else:
                # 未启用全局阈值：使用模型的最优阈值
                thresholds[model_name] = get_model_threshold(model_name)

    # 根据判断方法计算结果
    if method == "加权平均":
        # 加权平均
        total_weight = sum(model_weights.get(m, 1.0) for m in selected_models)
        weighted_score = sum(scores[m] * model_weights.get(m, 1.0)
                            for m in selected_models) / total_weight
        final_label = "不安全" if weighted_score >= global_threshold else "安全"

        detail_lines = [
            f"{'🔴' if final_label == '不安全' else '🟢'} **{final_label}**\n",
            f"**加权平均分数**：`{weighted_score:.4f}`（阈值：`{global_threshold}`）\n",
            f"**判断方法**：加权平均\n",
            f"**使用模型数**：{len(scores)}\n",
            "---\n",
            "**各模型详情**：\n"
        ]

        for model_name, score in scores.items():
            weight = model_weights.get(model_name, 1.0)
            threshold = thresholds[model_name]
            model_label = "不安全" if score >= threshold else "安全"
            locked = "🔒" if model_name in locked_models else ""
            detail_lines.append(
                f"- {model_name} {locked}: `{score:.4f}` (权重: {weight:.1f}, 阈值: {threshold:.2f}) → {model_label}"
            )

    else:  # 投票法
        # 每个模型根据自己的阈值投票
        votes = {}
        for model_name, score in scores.items():
            threshold = thresholds[model_name]
            votes[model_name] = "不安全" if score >= threshold else "安全"

        # 统计投票
        unsafe_count = sum(1 for v in votes.values() if v == "不安全")
        safe_count = len(votes) - unsafe_count
        final_label = "不安全" if unsafe_count > safe_count else "安全"

        detail_lines = [
            f"{'🔴' if final_label == '不安全' else '🟢'} **{final_label}**\n",
            f"**投票结果**：不安全 {unsafe_count} 票，安全 {safe_count} 票\n",
            f"**判断方法**：投票法（少数服从多数）\n",
            f"**使用模型数**：{len(scores)}\n",
            "---\n",
            "**各模型投票**：\n"
        ]

        for model_name, score in scores.items():
            threshold = thresholds[model_name]
            vote = votes[model_name]
            locked = "🔒" if model_name in locked_models else ""
            detail_lines.append(
                f"- {model_name} {locked}: `{score:.4f}` (阈值: {threshold:.2f}) → {vote}"
            )

    detail_lines.append(f"\n---\n{'图像疑似被篡改/伪造' if final_label == '不安全' else '图像真实可信'}")

    return final_label, "\n".join(detail_lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="图像伪造检测") as demo:
        gr.Markdown(
            """
            # 🔍 面部伪造检测平台
            上传图像以检测是否被篡改或伪造。
            支持多模型联合判断，提供加权平均和投票法两种方式。
            """
        )

        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                model_selector = gr.CheckboxGroup(
                    choices=AVAILABLE_MODELS,
                    value=[DEFAULT_MODEL] if DEFAULT_MODEL else [],
                    label="选择模型（可多选）",
                    info="选择一个或多个模型进行联合判断"
                )

                method_radio = gr.Radio(
                    choices=["加权平均", "投票法"],
                    value="加权平均",
                    label="判断方法"
                )

                # 动态说明文字（模拟 info 样式）
                method_info = gr.HTML(
                    value='<p style="margin-top: -12px; margin-bottom: 12px; font-size: 0.85em; color: #666; line-height: 1.4;">根据每个模型的权重计算加权平均分数，然后与全局阈值比较得出最终判断</p>'
                )

                # 全局阈值开关
                enable_global_threshold = gr.Checkbox(
                    label="启用全局阈值",
                    value=False,
                    info="关闭时各模型使用独立最优阈值，开启后使用下方全局阈值"
                )

                global_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="全局检测阈值",
                    info="应用于所有未锁定的模型",
                    interactive=False
                )

                run_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

                # 高级设置（渐进式披露）
                with gr.Accordion("⚙️ 高级设置", open=False):
                    gr.Markdown("### 模型权重与独立阈值")
                    gr.Markdown("💡 提示：权重仅在加权平均模式下生效；锁定后模型将使用独立阈值")

                    # 为每个模型创建权重、阈值和锁定控件
                    model_weights = {}
                    model_thresholds = {}
                    model_locks = {}

                    for model_name in AVAILABLE_MODELS:
                        optimal_threshold = get_model_threshold(model_name)

                        with gr.Row():
                            gr.Markdown(f"**{model_name}**")
                            model_weights[model_name] = gr.Slider(
                                minimum=0.1,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                label="权重",
                                scale=2
                            )
                            model_thresholds[model_name] = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=optimal_threshold,
                                step=0.001,
                                label="独立阈值",
                                scale=2
                            )
                            model_locks[model_name] = gr.Checkbox(
                                label="锁定",
                                value=False,
                                scale=1
                            )

            # 右侧：图像上传和结果显示
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="上传图像")

                label_output = gr.Label(label="预测结果")
                detail_output = gr.Markdown(label="详细信息")

        # 准备输入参数
        weight_inputs = [model_weights[m] for m in AVAILABLE_MODELS]
        threshold_inputs = [model_thresholds[m] for m in AVAILABLE_MODELS]
        lock_inputs = [model_locks[m] for m in AVAILABLE_MODELS]

        # 事件处理函数
        def update_method_info(method):
            """更新判断方法说明"""
            if method == "加权平均":
                text = "根据每个模型的权重计算加权平均分数，然后与全局阈值比较得出最终判断"
            else:
                text = "每个模型根据自己的阈值独立判断，然后少数服从多数，得票多的结果为最终判断"

            return f'<p style="margin-top: -12px; margin-bottom: 12px; font-size: 0.85em; color: #666; line-height: 1.4;">{text}</p>'

        # 绑定事件
        method_radio.change(
            fn=update_method_info,
            inputs=[method_radio],
            outputs=[method_info]
        )

        # 启用全局阈值时，激活滑块；关闭时，禁用滑块
        enable_global_threshold.change(
            fn=lambda enabled: gr.update(interactive=enabled),
            inputs=[enable_global_threshold],
            outputs=[global_threshold]
        )

        # 全局阈值改变时，同步到未锁定的模型
        def sync_global_to_individual(global_thresh, enable_global, *locks):
            """当全局阈值启用时，同步到未锁定的模型"""
            if not enable_global:
                return [gr.update() for _ in locks]

            updates = []
            for is_locked in locks:
                if not is_locked:
                    updates.append(gr.update(value=global_thresh))
                else:
                    updates.append(gr.update())
            return updates

        global_threshold.change(
            fn=sync_global_to_individual,
            inputs=[global_threshold, enable_global_threshold] + lock_inputs,
            outputs=threshold_inputs
        )

        def predict_wrapper(image, selected_models, method, enable_global, global_thresh, *args):
            # 解析权重、阈值和锁定状态
            n_models = len(AVAILABLE_MODELS)
            weights = args[:n_models]
            thresholds = args[n_models:2*n_models]
            locks = args[2*n_models:]

            # 构建字典
            weight_dict = {AVAILABLE_MODELS[i]: weights[i] for i in range(n_models)}
            threshold_dict = {AVAILABLE_MODELS[i]: thresholds[i] for i in range(n_models)}
            locked_list = [AVAILABLE_MODELS[i] for i in range(n_models) if locks[i]]

            return predict(image, selected_models, method, global_thresh, enable_global,
                         weight_dict, threshold_dict, locked_list)

        run_btn.click(
            fn=predict_wrapper,
            inputs=[img_input, model_selector, method_radio, enable_global_threshold, global_threshold] +
                   weight_inputs + threshold_inputs + lock_inputs,
            outputs=[label_output, detail_output],
        )

        gr.HTML("""
            <div style="text-align: center; padding: 16px 0 8px; font-size: 0.9em;">
                <a href="https://github.com/Q1ngS0ng" target="_blank"
                   style="color: #666; text-decoration: none; margin-right: 20px; display: inline-flex; align-items: center; gap: 6px;">
                    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    YuwanZ
                </a>
                <a href="https://q1ngs0ng.github.io/" target="_blank"
                   style="color: #666; text-decoration: none; display: inline-flex; align-items: center; gap: 6px;">
                    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zM1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0z"/>
                        <path d="M8 3.5a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5H4a.5.5 0 0 1 0-1h3.5V4a.5.5 0 0 1 .5-.5z"/>
                        <path d="M8 0a8 8 0 0 1 8 8 .5.5 0 0 1-1 0A7 7 0 1 0 8 15a.5.5 0 0 1 0 1A8 8 0 0 1 8 0z"/>
                    </svg>
                    YuwanZ
                </a>
            </div>
        """)

    return demo


if __name__ == "__main__":
    print(f"📦 模型目录: {LOCAL_MODELS_DIR}")
    print(f"🔍 使用本地模型: {USE_LOCAL}")
    print(f"📋 可用模型: {AVAILABLE_MODELS}")
    print(f"🎯 默认模型: {DEFAULT_MODEL}")
    print(f"⚙️  启动信息: {_startup_msg}")
    print("")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
