#!/bin/bash

# ============================================
# 垃圾分类 - ResNet18 + Grad-CAM 本地实验
# ============================================

echo "=========================================="
echo "垃圾分类 - ResNet18 + Grad-CAM 本地实验"
echo "=========================================="
echo ""

# ============================================
# 激活 conda 环境
# ============================================
source ~/anaconda3/etc/profile.d/conda.sh  # 请根据实际路径调整
conda activate resnet_gradcam_env

if [ $? -ne 0 ]; then
    echo "❌ 无法激活 conda 环境！"
    echo "请先运行: conda env create -f environment.yml"
    exit 1
fi

echo "✅ Conda 环境已激活"
echo ""

# ============================================
# 设置 Python 路径
# ============================================
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/model/ResNet18"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/model/ResNet18/code"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/model/GradCAM"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/model/CNN"

echo "✅ PYTHONPATH 已设置"
echo ""

# ============================================
# 创建必要的 __init__.py 文件
# ============================================
echo "创建 __init__.py 文件..."

if [ ! -f "model/__init__.py" ]; then touch model/__init__.py; fi
if [ ! -f "model/ResNet18/__init__.py" ]; then touch model/ResNet18/__init__.py; fi
if [ ! -f "model/ResNet18/code/__init__.py" ]; then touch model/ResNet18/code/__init__.py; fi
if [ ! -f "model/GradCAM/__init__.py" ]; then touch model/GradCAM/__init__.py; fi
if [ ! -f "model/CNN/__init__.py" ]; then touch model/CNN/__init__.py; fi

echo "✅ __init__.py 文件已创建"
echo ""

# ============================================
# 创建输出目录
# ============================================
echo "创建输出目录..."

mkdir -p logs
mkdir -p model/ResNet18/code/checkpoints
mkdir -p model/ResNet18/outputs
mkdir -p model/GradCAM/outputs
mkdir -p results

echo "✅ 输出目录已创建"
echo ""

# ============================================
# 设置日志文件
# ============================================
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo ""

# ============================================
# 检查必要文件
# ============================================
echo "检查项目文件..."

MISSING_FILES=0

if [ ! -f "model/ResNet18/code/main.py" ]; then
    echo "❌ 错误: model/ResNet18/code/main.py 不存在！"
    MISSING_FILES=1
fi

if [ ! -f "model/GradCAM/GradCAM_CNN.py" ]; then
    echo "❌ 错误: model/GradCAM/GradCAM_CNN.py 不存在！"
    MISSING_FILES=1
fi

if [ ! -d "data_split/train" ]; then
    echo "⚠️ 警告: data_split/train 不存在，请检查数据集！"
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "❌ 缺少必要文件，请检查项目结构！"
    exit 1
fi

echo "✅ 文件检查通过"
echo ""

# ============================================
# 开始运行实验
# ============================================
echo "==========================================" | tee -a "$LOG_FILE"
echo "ResNet + Grad-CAM 实验日志" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================
# 实验 1: ResNet18 训练
# ============================================
echo "==========================================" | tee -a "$LOG_FILE"
echo "[1/2] ResNet18 垃圾分类训练" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd model/ResNet18/code
echo "开始运行 ResNet18 实验..." | tee -a "../../../$LOG_FILE"
python main.py 2>&1 | tee -a "../../../$LOG_FILE"
RESNET_EXIT_CODE=${PIPESTATUS[0]}
cd ../../..

if [ $RESNET_EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet18 实验完成！" | tee -a "$LOG_FILE"
else
    echo "❌ ResNet18 实验失败！" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ============================================
# 实验 2: Grad-CAM 可视化
# ============================================
echo "==========================================" | tee -a "$LOG_FILE"
echo "[2/2] Grad-CAM 可视化" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd model/GradCAM

# 检查模型文件
if [ ! -f "../ResNet18/code/checkpoints/best_model.pth" ] && [ ! -f "../ResNet18/code/best_model.pth" ]; then
    echo "⚠️ 警告: ResNet18 模型文件不存在！" | tee -a "../../$LOG_FILE"
    echo "将使用默认路径尝试加载" | tee -a "../../$LOG_FILE"
fi

echo "开始运行 Grad-CAM 实验..." | tee -a "../../$LOG_FILE"
python GradCAM_CNN.py \
    --ckpt ../ResNet18/code/checkpoints/best_model.pth \
    --split-root ../../data_split \
    --image-size 224 \
    --alpha 0.45 \
    --out-dir outputs 2>&1 | tee -a "../../$LOG_FILE"

GRADCAM_EXIT_CODE=${PIPESTATUS[0]}
cd ../..

if [ $GRADCAM_EXIT_CODE -eq 0 ]; then
    echo "✅ Grad-CAM 实验完成！" | tee -a "$LOG_FILE"
else
    echo "❌ Grad-CAM 实验失败！" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# ============================================
# 完成
# ============================================
echo "==========================================" | tee -a "$LOG_FILE"
echo "✅ 所有实验完成！" | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo ""
echo "=========================================="
echo "✅ 所有实验执行完成！"
echo "=========================================="
echo ""
echo "📁 结果位置:"
echo "  - ResNet18 模型: model/ResNet18/code/checkpoints/"
echo "  - ResNet18 结果: model/ResNet18/outputs/"
echo "  - Grad-CAM 结果: model/GradCAM/outputs/"
echo "  - 日志文件: $LOG_FILE"
echo ""
echo "📝 Kaggle 上运行的 CNN 实验:"
echo "  - 请访问: https://www.kaggle.com/code/kageyamafirst/garbage-classification-cnn/edit"
echo "  - Kaggle 代码目录: kaggle_CNN/"
echo ""