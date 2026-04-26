#!/bin/bash
# 使用4张GPU并行重新评估所有模型

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/scripts/evaluation"
cd "$SCRIPT_DIR"

MODELS=("convnext_base" "dinov2_base" "fastervit_2" "inceptionnext_base" "internvit_300m" "mambavision_t" "maxvit_base")
GPUS=(0 1 2 3)

echo "🚀 使用 ${#GPUS[@]} 张GPU并行重新评估 ${#MODELS[@]} 个模型"
echo ""

# 分配模型到GPU
declare -A gpu_models
for i in "${!MODELS[@]}"; do
    gpu_id=${GPUS[$((i % ${#GPUS[@]}))]}
    if [ -z "${gpu_models[$gpu_id]}" ]; then
        gpu_models[$gpu_id]="${MODELS[$i]}"
    else
        gpu_models[$gpu_id]="${gpu_models[$gpu_id]} ${MODELS[$i]}"
    fi
done

mkdir -p results/evaluate/logs

# 启动评估任务
pids=()
for gpu_id in "${!gpu_models[@]}"; do
    models_list="${gpu_models[$gpu_id]}"
    echo "GPU $gpu_id: $models_list"

    (
        for model in $models_list; do
            echo "[GPU $gpu_id] 评估 $model..."
            CUDA_VISIBLE_DEVICES=$gpu_id python "${EVAL_DIR}/evaluate.py" --model $model --batch_size 16
        done
    ) > "results/evaluate/logs/gpu${gpu_id}_reeval.log" 2>&1 &

    pids+=($!)
done

echo ""
echo "等待所有评估完成..."
echo "进程 PIDs: ${pids[@]}"

# 等待所有任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "✅ 所有模型重新评估完成！"
echo ""
echo "🔍 运行阈值优化..."
python "${EVAL_DIR}/optimize_thresholds.py"

echo ""
echo "✅ 阈值优化完成！"
