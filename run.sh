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
export PYTHONPATH="${PYTHONPATH}:$(pwd)/model/CNN/code"

echo "✅ PYTHONPATH 已设置"
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

# 检查 ResNet18 训练文件
if [ -f "model/ResNet18/code/resnet_garbage_classification.py" ]; then
    echo "✅ resnet_garbage_classification.py 存在"
    TRAIN_FILE="resnet_garbage_classification.py"
elif [ -f "model/ResNet18/code/main.py" ] && [ -s "model/ResNet18/code/main.py" ]; then
    echo "✅ main.py 存在"
    TRAIN_FILE="main.py"
else
    echo "❌ 错误: 找不到 ResNet18 训练文件！"
    MISSING_FILES=1
fi

# 检查 GradCAM
if [ -f "model/GradCAM/GradCAM_CNN.py" ]; then
    echo "✅ GradCAM_CNN.py 存在"
else
    echo "❌ 错误: model/GradCAM/GradCAM_CNN.py 不存在！"
    MISSING_FILES=1
fi

# 检查数据集
if [ -d "data_split/train" ]; then
    echo "✅ data_split/train 存在"
else
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

echo "使用训练脚本: $TRAIN_FILE" | tee -a "../../../$LOG_FILE"
echo "开始运行 ResNet18 实验..." | tee -a "../../../$LOG_FILE"

python "$TRAIN_FILE" 2>&1 | tee -a "../../../$LOG_FILE"

RESNET_EXIT_CODE=${PIPESTATUS[0]}
cd ../../..

if [ $RESNET_EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet18 实验完成！" | tee -a "$LOG_FILE"
else
    echo "❌ ResNet18 实验失败！(退出码: $RESNET_EXIT_CODE)" | tee -a "$LOG_FILE"
    echo "请检查日志文件: $LOG_FILE"
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

# 查找模型权重
CKPT_PATH=""
if [ -f "../ResNet18/code/checkpoints/best_model.pth" ]; then
    CKPT_PATH="../ResNet18/code/checkpoints/best_model.pth"
elif [ -f "../ResNet18/code/best_model.pth" ]; then
    CKPT_PATH="../ResNet18/code/best_model.pth"
elif [ -f "../ResNet18/outputs/best_model.pth" ]; then
    CKPT_PATH="../ResNet18/outputs/best_model.pth"
else
    echo "⚠️ 警告: 未找到 ResNet18 模型权重！" | tee -a "../../$LOG_FILE"
    echo "将使用随机权重运行 Grad-CAM" | tee -a "../../$LOG_FILE"
    CKPT_PATH=""
fi

echo "开始运行 Grad-CAM 实验..." | tee -a "../../$LOG_FILE"

if [ -n "$CKPT_PATH" ]; then
    echo "使用模型: $CKPT_PATH" | tee -a "../../$LOG_FILE"
    python GradCAM_CNN.py \
        --ckpt "$CKPT_PATH" \
        --split-root ../../data_split \
        --image-size 224 \
        --alpha 0.45 \
        --out-dir outputs 2>&1 | tee -a "../../$LOG_FILE"
else
    python GradCAM_CNN.py \
        --split-root ../../data_split \
        --image-size 224 \
        --alpha 0.45 \
        --out-dir outputs 2>&1 | tee -a "../../$LOG_FILE"
fi

GRADCAM_EXIT_CODE=${PIPESTATUS[0]}
cd ../..

if [ $GRADCAM_EXIT_CODE -eq 0 ]; then
    echo "✅ Grad-CAM 实验完成！" | tee -a "$LOG_FILE"
else
    echo "❌ Grad-CAM 实验失败！(退出码: $GRADCAM_EXIT_CODE)" | tee -a "$LOG_FILE"
    echo "请检查日志文件: $LOG_FILE"
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