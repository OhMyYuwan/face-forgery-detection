#!/bin/bash

# conda 环境名称
CONDA_ENV="simswap"

# ============== 用户配置区域 ==============
# GPU列表 (用空格分隔)
GPU_IDS=("3")

# 数据集配置
DATASET_ROOT="your_dataset_path"
DEEPFAKE_METHODS="your_deepfake_types"  # "all" 或逗号分隔的方法列表，如 "progan,stylegan2,vqgan"
SAVE_PATH="your_save_path"

# 模型配置
BACKBONE="your_back_bone"

# Stage 1 配置 (表示学习)
STAGE1_EPOCHS=200
STAGE1_BATCH_SIZE=32
STAGE1_LR=5e-5
STAGE1_HEAD_LR=0.01
STAGE1_MOMENTUM=0.9
STAGE1_WEIGHT_DECAY=1e-4

# Stage 2 配置 (伪造检测)
STAGE2_EPOCHS=100
STAGE2_BATCH_SIZE=16
STAGE2_LR=0.01
STAGE2_MOMENTUM=0.9
STAGE2_WEIGHT_DECAY=1e-4
LAMBDA_AUX=0.3  # 辅助对比损失权重

# 训练配置
NUM_WORKERS=4
TEMPERATURE=0.1
IMG_SIZE=224
EVAL_INTERVAL=10  # 评估间隔（每N个epoch评估一次）

# 学习率调度
COSINE=false  # 使用余弦退火
WARMUP_FROM=0.01
WARMUP_EPOCHS=0
LR_DECAY=0.1

# 可选：预训练的 Stage 1 模型路径（留空则从头训练）
STAGE1_MODEL=""
# ===========================================

# 训练脚本路径
TRAIN_SCRIPT="train_face_forgery.py"

# 初始化并激活 conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

conda activate $CONDA_ENV
echo "[Conda] Activated environment: $CONDA_ENV"

# 验证配置
NUM_GPUS=${#GPU_IDS[@]}

# 创建日志目录
mkdir -p log

echo "========== xinyuan_fine Face Forgery Detection Training =========="
echo "Script: $TRAIN_SCRIPT"
echo "GPUs: ${GPU_IDS[@]}"
echo "Dataset: $DATASET_ROOT"
echo "Deepfake Methods: $DEEPFAKE_METHODS"
echo "Backbone: $BACKBONE"
echo ""
echo "Stage 1 Config:"
echo "  - Epochs: $STAGE1_EPOCHS"
echo "  - Batch Size: $STAGE1_BATCH_SIZE"
echo "  - Learning Rate: $STAGE1_LR"
echo "  - Optimizer: SGD (momentum=$STAGE1_MOMENTUM, weight_decay=$STAGE1_WEIGHT_DECAY)"
echo ""
echo "Stage 2 Config:"
echo "  - Epochs: $STAGE2_EPOCHS"
echo "  - Batch Size: $STAGE2_BATCH_SIZE"
echo "  - Learning Rate: $STAGE2_LR"
echo "  - Lambda Aux: $LAMBDA_AUX"
echo "  - Samples Per Class: $SAMPLES_PER_CLASS"
echo "  - Eval Interval: $EVAL_INTERVAL epochs"
echo ""

# 启动训练
PIDS=()
for i in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    
    echo "[GPU $GPU_ID] Starting training..."
    
    # 构建命令
    CMD="python $TRAIN_SCRIPT \
        --gpu $GPU_ID \
        --dataset_root $DATASET_ROOT \
        --deepfake_methods $DEEPFAKE_METHODS \
        --savepath ${SAVE_PATH}/gpu${GPU_ID} \
        --backbone $BACKBONE \
        --stage1_epochs $STAGE1_EPOCHS \
        --stage1_batch_size $STAGE1_BATCH_SIZE \
        --stage1_lr $STAGE1_LR \
        --stage1_head_lr $STAGE1_HEAD_LR \
        --stage1_momentum $STAGE1_MOMENTUM \
        --stage1_weight_decay $STAGE1_WEIGHT_DECAY \
        --stage2_epochs $STAGE2_EPOCHS \
        --stage2_batch_size $STAGE2_BATCH_SIZE \
        --stage2_lr $STAGE2_LR \
        --stage2_momentum $STAGE2_MOMENTUM \
        --stage2_weight_decay $STAGE2_WEIGHT_DECAY \
        --lambda_aux $LAMBDA_AUX \
        --num_workers $NUM_WORKERS \
        --temperature $TEMPERATURE \
        --img_size $IMG_SIZE \
        --eval_interval $EVAL_INTERVAL \
        --warmup_from $WARMUP_FROM \
        --warmup_epochs $WARMUP_EPOCHS \
        --lr_decay $LR_DECAY"
    
    # 添加 cosine 标志
    if [ "$COSINE" = true ]; then
        CMD="$CMD --cosine"
    fi
    
    # 添加 Stage 1 模型路径（如果提供）
    if [ -n "$STAGE1_MODEL" ]; then
        CMD="$CMD --stage1_model $STAGE1_MODEL"
    fi
    
    # 后台运行
    nohup $CMD > log/xinyuan_MambaVision_gpu${GPU_ID}.log 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "  - PID: $PID"
    echo "  - Log: log/xinyuan_MambaVision_gpu${GPU_ID}.log"
    echo ""
    sleep 3
done

# 训练信息汇总
echo "========== All trainings started =========="
echo ""
echo "Process IDs:"
for i in "${!PIDS[@]}"; do
    echo "  - GPU ${GPU_IDS[$i]}: PID ${PIDS[$i]}"
done
echo ""
echo "Training configs:"
echo "  - Dataset: $DATASET_ROOT"
echo "  - Backbone: $BACKBONE"
echo "  - Stage1: $STAGE1_EPOCHS epochs, batch=$STAGE1_BATCH_SIZE, lr=$STAGE1_LR"
echo "  - Stage2: $STAGE2_EPOCHS epochs, batch=$STAGE2_BATCH_SIZE, lr=$STAGE2_LR"
echo "  - Evaluation: Every $EVAL_INTERVAL epochs"
echo ""
echo "Monitor progress:"
for i in "${!GPU_IDS[@]}"; do
    echo "  - GPU ${GPU_IDS[$i]}: tail -f log/xinyuan_MambaVision_gpu${GPU_IDS[$i]}.log"
done
echo ""
echo "Monitor all logs:"
echo "  tail -f log/xinyuan_MambaVision_gpu*.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill all trainings:"
echo "  kill ${PIDS[@]}"
echo ""
echo "Expected output files:"
echo "  - Stage 1 models: ${SAVE_PATH}/gpu*/stage1_models_${BACKBONE}/"
echo "  - Stage 2 models: ${SAVE_PATH}/gpu*/stage2_detnet_enhance/path_models_${BACKBONE}/"
echo "  - Best model (accuracy): best_model_acc.pth"
echo "  - Best model (loss): best_model.pth"
