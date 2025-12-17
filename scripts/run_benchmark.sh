#!/bin/bash

# ==============================================================================
# Usage:
#   bash scripts/run_benchmark.sh --task rosenbrock --repeats 1 --gpu 0
#   bash scripts/run_benchmark.sh --task stybtang --repeats 3 --gpu 1
# ==============================================================================

TASK=""
REPEATS=1
GPU_ID=0
START_SEED=42

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift ;;
        --repeats) REPEATS="$2"; shift ;;
        --gpu) GPU_ID="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$TASK" ]]; then
    echo "Error: Please specify --task (rosenbrock or stybtang)"
    exit 1
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "=== Starting Benchmark: $TASK (Repeats: $REPEATS, GPU: $GPU_ID) ==="

# 定义该任务下的三种算法配置 (Template Paths)
# 注意：这里假设配置文件都在 configs/benchmark/ 目录下
CONFIG_DIR="configs/benchmark"
declare -a ALGOS=("adaptive" "oracle" "standard")

# 确保目录存在
mkdir -p logs

for ALGO in "${ALGOS[@]}"; do
    TEMPLATE_CFG="$CONFIG_DIR/${TASK}_${ALGO}.yaml"
    
    if [[ ! -f "$TEMPLATE_CFG" ]]; then
        echo "Error: Config file $TEMPLATE_CFG not found!"
        continue
    fi

    echo "--------------------------------------------------------"
    echo ">>> Running Algorithm: $ALGO"
    echo "--------------------------------------------------------"

    for ((i=0; i<REPEATS; i++)); do
        CURRENT_SEED=$((START_SEED + i))
        
        # 1. 生成临时配置文件
        # 我们需要修改 YAML 中的 seed 和 name 字段
        # 假设 YAML 中格式为 'seed: 42' 和 'name: "ExpName"'
        TEMP_CFG="${TEMPLATE_CFG%.yaml}_run_seed${CURRENT_SEED}.yaml"
        
        # 使用 sed 替换 (适配 Linux)
        # 1. 替换 seed: 42 -> seed: CURRENT_SEED
        # 2. 替换 name: "Exp" -> name: "Exp_SeedX"
        sed "s/seed: .*/seed: $CURRENT_SEED/" "$TEMPLATE_CFG" > "$TEMP_CFG"
        # 这里做一个简单的替换，在 name 的引号结束前插入 _SeedX
        sed -i "s/name: \"\(.*\)\"/name: \"\1_Seed${CURRENT_SEED}\"/" "$TEMP_CFG"
        
        echo "   [Run $i/$REPEATS] Seed: $CURRENT_SEED | Config: $TEMP_CFG"
        
        # 2. 运行实验
        python scripts/run_experiment.py --config "$TEMP_CFG"
        
        # 3. 清理临时文件
        rm "$TEMP_CFG"
        
        echo "   [Done] Seed $CURRENT_SEED finished."
        echo ""
    done
done

# ==============================================================================
# 自动绘图
# ==============================================================================
echo "=== All Runs Finished. Generating Plot... ==="

# 根据 Task 定义图表标题和日志模式
if [[ "$TASK" == "rosenbrock" ]]; then
    TITLE="Sum-Rosenbrock 50D Comparison"
    PATTERNS=( "logs/Rosenbrock_50D_Adaptive_*" "logs/Rosenbrock_50D_Oracle_*" "logs/Rosenbrock_50D_Standard_*" )
    LABELS=( "Adaptive BO" "Oracle BO" "Standard BO" )
elif [[ "$TASK" == "stybtang" ]]; then
    TITLE="Styblinski-Tang 100D Comparison"
    PATTERNS=( "logs/Stybtang_100D_Adaptive_*" "logs/Stybtang_100D_Oracle_*" "logs/Stybtang_100D_Standard_*" )
    LABELS=( "Adaptive BO" "Oracle BO" "Standard BO" )
fi

# 调用绘图脚本
python scripts/plot_results.py \
    --labels "${LABELS[@]}" \
    --patterns "${PATTERNS[@]}" \
    --title "$TITLE" \
    --output "${TASK}_benchmark_result.png"

echo "=== Benchmark Complete! Check ${TASK}_benchmark_result.png ==="
