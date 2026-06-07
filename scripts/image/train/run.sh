#!/bin/bash

CONDA_ENV="simswap"

# ============== User config ==============
GPU_IDS=("your_gpu_id")
DATASET_ROOT="your_dataset_root"
DEEPFAKE_METHODS="your_deepfake_methods"  # use "all" or comma-separated methods
SAVE_PATH="your_save_path"
BACKBONE="your_model"

# Stage 1: representation learning. Set to 0 when using a pretrained Stage 1 model.
STAGE1_EPOCHS=50
STAGE1_BATCH_SIZE=12
STAGE1_LR=1e-5
STAGE1_HEAD_LR=0.01
STAGE1_MOMENTUM=0.9
STAGE1_WEIGHT_DECAY=1e-4
STAGE1_MODEL=""  # set to your_stage1_checkpoint to skip/reuse Stage 1

# Stage 2: core feature selection.
STAGE2_EPOCHS=20
STAGE2_BATCH_SIZE=4
STAGE2_LR=1e-4
STAGE2_LR_BACKBONE=1e-6
STAGE2_UPDATE_BACKBONE=false
STAGE2_MOMENTUM=0.9
STAGE2_WEIGHT_DECAY=1e-4
SELECTION_CONTRAST_WEIGHT=1.0
SELECTION_REAL_COMPACT_WEIGHT=0.2
SELECTION_REAL_VIEW_WEIGHT=0.2
SELECTION_FAKE_PROTO_WEIGHT=1.5
SELECTION_RETAIN_WEIGHT=0.05

# Stage 3: forgery detection.
STAGE3_EPOCHS=100
STAGE3_LR=0.001
STAGE3_LR_BACKBONE=1e-5
STAGE3_MOMENTUM=0.9
STAGE3_WEIGHT_DECAY=1e-4
LAMBDA_AUX=0.3

NUM_WORKERS=4
TEMPERATURE=0.1
IMG_SIZE=224
EVAL_INTERVAL=10
COSINE=false
WARMUP_FROM=0.01
WARMUP_EPOCHS=0
LR_DECAY=0.1
# =========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

conda activate "$CONDA_ENV"
echo "[Conda] Activated environment: $CONDA_ENV"

mkdir -p log
TRAIN_SCRIPT="train_face_forgery.py"

PLACEHOLDER_VALUES=(
    "your_gpu_id"
    "your_dataset_root"
    "your_deepfake_methods"
    "your_save_path"
    "your_model"
)
for PLACEHOLDER in "${PLACEHOLDER_VALUES[@]}"; do
    if [[ " ${GPU_IDS[*]} $DATASET_ROOT $DEEPFAKE_METHODS $SAVE_PATH $BACKBONE " == *" $PLACEHOLDER "* ]]; then
        echo "[Config Error] Please replace placeholder: $PLACEHOLDER"
        echo "Edit this script before running training."
        exit 1
    fi
done

echo "========== image2 Three-Stage Core-Select Training =========="
echo "Script: $TRAIN_SCRIPT"
echo "GPUs: ${GPU_IDS[@]}"
echo "Dataset: $DATASET_ROOT"
echo "Methods: $DEEPFAKE_METHODS"
echo "Backbone: $BACKBONE"
echo "Save path: $SAVE_PATH"
echo ""

PIDS=()
for GPU_ID in "${GPU_IDS[@]}"; do
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
        --stage2_lr_backbone $STAGE2_LR_BACKBONE \
        --stage2_momentum $STAGE2_MOMENTUM \
        --stage2_weight_decay $STAGE2_WEIGHT_DECAY \
        --selection_contrast_weight $SELECTION_CONTRAST_WEIGHT \
        --selection_real_compact_weight $SELECTION_REAL_COMPACT_WEIGHT \
        --selection_real_view_weight $SELECTION_REAL_VIEW_WEIGHT \
        --selection_fake_proto_weight $SELECTION_FAKE_PROTO_WEIGHT \
        --selection_retain_weight $SELECTION_RETAIN_WEIGHT \
        --stage3_epochs $STAGE3_EPOCHS \
        --stage3_lr $STAGE3_LR \
        --stage3_lr_backbone $STAGE3_LR_BACKBONE \
        --stage3_momentum $STAGE3_MOMENTUM \
        --stage3_weight_decay $STAGE3_WEIGHT_DECAY \
        --lambda_aux $LAMBDA_AUX \
        --num_workers $NUM_WORKERS \
        --temperature $TEMPERATURE \
        --img_size $IMG_SIZE \
        --eval_interval $EVAL_INTERVAL \
        --warmup_from $WARMUP_FROM \
        --warmup_epochs $WARMUP_EPOCHS \
        --lr_decay $LR_DECAY"

    if [ "$COSINE" = true ]; then
        CMD="$CMD --cosine"
    fi
    if [ "$STAGE2_UPDATE_BACKBONE" = true ]; then
        CMD="$CMD --stage2_update_backbone"
    fi
    if [ -n "$STAGE1_MODEL" ]; then
        CMD="$CMD --stage1_model $STAGE1_MODEL"
    fi

    LOG_FILE="log/${BACKBONE}_core_select_gpu${GPU_ID}.log"
    echo "[GPU $GPU_ID] Starting training"
    echo "  Log: $SCRIPT_DIR/$LOG_FILE"
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
    echo "  PID: ${PIDS[-1]}"
    sleep 3
done

echo ""
echo "========== All trainings started =========="
for i in "${!GPU_IDS[@]}"; do
    echo "GPU ${GPU_IDS[$i]}: PID ${PIDS[$i]}"
done
echo "Monitor: tail -f $SCRIPT_DIR/log/${BACKBONE}_core_select_gpu*.log"
echo "Kill: kill ${PIDS[@]}"
