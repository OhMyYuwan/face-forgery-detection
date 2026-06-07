#!/bin/bash
# Face Forgery Detection Platform - Service Manager
# Usage: ./start.sh {start|stop|restart|status|help}

set -e

# Configuration
PORT=7860
APP_NAME="Face Forgery Detection"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_DIR="${APP_DIR}/Space"
APP_SCRIPT="${SPACE_DIR}/app.py"
PID_FILE="${SPACE_DIR}/.app.pid"
LOG_FILE="${SPACE_DIR}/app.log"

# IMPORTANT: dofnet / sida / aeroblade / safe were developed under `sidaend`.
# We now use sidaend as the unified environment. If older OurNet models break
# under sidaend, install missing deps in sidaend (timm, huggingface-hub, gradio,
# etc.) rather than switching back.
CONDA_ENV="sidaend"

# Model weights directory
WEIGHTS_DIR="${APP_DIR}/OhMyYuwan/face-forgery-detection"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() { echo -e "${BLUE}ℹ${NC}  $1"; }
print_success() { echo -e "${GREEN}✅${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠️${NC}  $1"; }
print_error() { echo -e "${RED}❌${NC} $1"; }

# Check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Scan available models
# A valid model dir must contain model.py and pytorch_model.bin
# (or a symlink to a checkpoint, or an empty marker file for adapter-style
#  models like sida/aeroblade where weights live elsewhere on disk).
scan_models() {
    if [ ! -d "$WEIGHTS_DIR" ]; then
        return 1
    fi

    AVAILABLE_MODELS=()
    for dir in "$WEIGHTS_DIR"/*/; do
        [ ! -d "$dir" ] && continue
        model_name=$(basename "$dir")

        [[ "$model_name" == .* ]] && continue

        has_model_py=false
        has_weights=false

        [ -f "${dir}model.py" ] && has_model_py=true
        ( [ -f "${dir}pytorch_model.bin" ] || [ -L "${dir}pytorch_model.bin" ] ) && has_weights=true

        if $has_model_py && $has_weights; then
            AVAILABLE_MODELS+=("$model_name")
        fi
    done
}

# Display models table — also shows whether the model is OurNet-style
# (has its own backbone weights inline) or adapter-style (weights live
#  outside this directory and pytorch_model.bin is a marker/symlink).
show_models_table() {
    if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
        print_error "未找到可用模型"
        return 1
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-5s %-20s %-10s %-12s %-15s\n" "序号" "模型名称" "model.py" "权重文件" "类型"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local idx=1
    for model in "${AVAILABLE_MODELS[@]}"; do
        local model_dir="${WEIGHTS_DIR}/${model}"
        local has_py="❌"
        local has_weight="❌"
        local kind="ournet"

        [ -f "${model_dir}/model.py" ] && has_py="✅"
        ( [ -f "${model_dir}/pytorch_model.bin" ] || [ -L "${model_dir}/pytorch_model.bin" ] ) && has_weight="✅"

        # Detect inference_type from config.json (jq optional, fallback to grep)
        if [ -f "${model_dir}/config.json" ]; then
            if command -v jq >/dev/null 2>&1; then
                local it
                it=$(jq -r '.inference_type // "ournet"' "${model_dir}/config.json" 2>/dev/null || echo "ournet")
                kind="$it"
            else
                if grep -q '"inference_type"' "${model_dir}/config.json" 2>/dev/null; then
                    kind=$(grep '"inference_type"' "${model_dir}/config.json" | head -n1 | sed -E 's/.*"inference_type"[^"]*"([^"]+)".*/\1/')
                fi
            fi
        fi

        printf "%-5s %-20s %-10s %-12s %-15s\n" "$idx" "$model" "$has_py" "$has_weight" "$kind"
        ((idx++))
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_success "共找到 ${#AVAILABLE_MODELS[@]} 个可用模型"
    echo ""
}

check_weights() {
    print_info "扫描模型目录: $WEIGHTS_DIR"
    if [ ! -d "$WEIGHTS_DIR" ]; then
        print_error "模型目录不存在: $WEIGHTS_DIR"
        return 1
    fi
    scan_models
    show_models_table
    if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
        return 1
    fi
    return 0
}

check_port() {
    if netstat -tlnp 2>/dev/null | grep -q ":${PORT}"; then
        return 1
    fi
    return 0
}

kill_port_processes() {
    PIDS=$(netstat -tlnp 2>/dev/null | grep ":${PORT}" | awk '{print $NF}' | cut -d'/' -f1)
    [ -z "$PIDS" ] && return 0

    print_warning "端口 ${PORT} 被以下进程占用:"
    for PID in $PIDS; do
        PROC_INFO=$(ps -p $PID -o pid,cmd --no-headers 2>/dev/null || echo "$PID <unknown>")
        echo "  - $PROC_INFO"
    done

    read -p "是否终止这些进程? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for PID in $PIDS; do
            kill -9 $PID 2>/dev/null && print_success "已终止进程 $PID" || print_warning "无法终止进程 $PID"
        done
        sleep 2
        return 0
    else
        print_info "用户取消操作"
        return 1
    fi
}

start_service() {
    echo "🚀 启动 ${APP_NAME}..."
    echo ""

    if is_running; then
        PID=$(cat "$PID_FILE")
        print_warning "服务已在运行 (PID: $PID)"
        return 1
    fi

    if ! check_weights; then
        return 1
    fi

    if ! check_port; then
        if ! kill_port_processes; then
            print_error "端口 ${PORT} 被占用，无法启动"
            return 1
        fi
    fi

    print_info "激活 ${CONDA_ENV} 环境..."
    source /root/anaconda3/etc/profile.d/conda.sh
    if ! conda activate "$CONDA_ENV" 2>/dev/null; then
        print_error "激活 conda 环境失败: $CONDA_ENV"
        print_warning "请确认该环境存在 (conda env list)"
        return 1
    fi

    print_info "检查依赖..."
    # Old OurNet models need: gradio, torch, timm, huggingface-hub
    # New adapter models additionally need: transformers, diffusers, lpips,
    #   bitsandbytes (for SIDA 4/8-bit), opencv-python (for SIDA preprocessing)
    if ! pip list 2>/dev/null | grep -E "gradio|torch|timm|huggingface-hub" > /dev/null; then
        print_warning "缺少基础依赖 (gradio/torch/timm/huggingface-hub)"
    fi
    if ! pip list 2>/dev/null | grep -E "diffusers" > /dev/null; then
        print_warning "缺少 diffusers (AeroBlade 需要)"
    fi
    if ! pip list 2>/dev/null | grep -E "^lpips " > /dev/null; then
        print_warning "缺少 lpips (AeroBlade 需要)"
    fi
    print_success "依赖检查完成"
    echo ""

    print_info "启动应用 (后台运行)..."
    print_info "日志文件: ${LOG_FILE}"
    print_info "访问地址: http://localhost:${PORT}"
    echo ""

    nohup python "$APP_SCRIPT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 3
    if is_running; then
        print_success "服务启动成功 (PID: $(cat $PID_FILE))"
        print_info "使用 './start.sh status' 查看状态"
        print_info "使用 './start.sh stop' 停止服务"
        return 0
    else
        print_error "服务启动失败，请查看日志: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_service() {
    echo "🛑 停止 ${APP_NAME}..."
    echo ""

    if ! is_running; then
        print_warning "服务未运行"
        rm -f "$PID_FILE"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    print_info "正在停止进程 (PID: $PID)..."
    kill "$PID" 2>/dev/null

    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ps -p "$PID" > /dev/null 2>&1; then
        print_warning "进程未响应，强制终止..."
        kill -9 "$PID" 2>/dev/null
    fi

    rm -f "$PID_FILE"
    print_success "服务已停止"
    return 0
}

restart_service() {
    echo "🔄 重启 ${APP_NAME}..."
    echo ""
    stop_service
    sleep 2
    start_service
}

show_status() {
    echo "📊 ${APP_NAME} 状态"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if is_running; then
        PID=$(cat "$PID_FILE")
        print_success "服务运行中"
        echo "  PID: $PID"
        echo "  端口: $PORT"
        echo "  日志: $LOG_FILE"
        echo ""
        print_info "进程信息:"
        ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | sed 's/^/  /'
        echo ""
        if [ -f "$LOG_FILE" ]; then
            print_info "最近日志 (最后 10 行):"
            tail -n 10 "$LOG_FILE" | sed 's/^/  /'
        fi
    else
        print_warning "服务未运行"
        if [ -f "$PID_FILE" ]; then
            print_info "清理残留 PID 文件..."
            rm -f "$PID_FILE"
        fi
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

show_help() {
    cat << EOF
${APP_NAME} - 服务管理脚本

用法:
  ./start.sh {start|stop|restart|status|help}

命令:
  start      启动服务 (后台运行)
  stop       停止服务
  restart    重启服务
  status     查看服务状态
  help       显示此帮助信息

配置:
  端口:       ${PORT}
  应用目录:   ${APP_DIR}
  日志文件:   ${LOG_FILE}
  PID 文件:   ${PID_FILE}
  Conda 环境: ${CONDA_ENV}

支持的模型类型:
  ournet     - 老的 OurNet 风格 (convnext_base, dinov2_base, ...)
  dofnet     - DoFNet (ResNet18)
  sida       - SIDA-13B (LLaVA-2 风格 VLM)
  aeroblade  - AeroBlade (训练自由, 3 VAE + LPIPS)
  safe       - SAFE (ResNet50 + DWT)

EOF
}

show_overview() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${APP_NAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_info "扫描模型目录: $WEIGHTS_DIR"
    if [ -d "$WEIGHTS_DIR" ]; then
        scan_models
        show_models_table
    else
        print_error "模型目录不存在: $WEIGHTS_DIR"
        echo ""
    fi
    show_help
}

case "${1:-}" in
    start)   start_service ;;
    stop)    stop_service ;;
    restart) restart_service ;;
    status)  show_status ;;
    help|--help|-h) show_help ;;
    "")      show_overview ;;
    *)
        print_error "无效的命令: ${1}"
        echo ""
        show_help
        exit 1
        ;;
esac
