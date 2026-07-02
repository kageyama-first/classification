# 基于 CNN 与 ResNet-18 的垃圾图像分类实验
> 基于 Kaggle Garbage Classification 数据集，对 CNN、ResNet-18、数据增强策略、学习率及损失函数进行了系统对比实验。


# 项目简介

本项目围绕垃圾图像分类（Garbage Classification）任务展开，基于 Kaggle 公开数据集，对不同卷积神经网络模型及训练策略进行了系统实验。

与仅关注最终准确率不同，本项目重点分析：

* 不同网络结构（SimpleCNN、AdvancedCNN、ResNet-18）的性能差异；
* ImageNet 预训练迁移学习的作用；
* 数据增强策略对不同模型的影响；
* 学习率与损失函数对模型收敛和泛化能力的影响；
* 模型错误模式、Grad-CAM 可解释性分析以及注意力机制的改进方向。


# 数据集

数据集采用 Kaggle 公开的 **Garbage Classification Dataset**。

## 数据集信息

* 图像总数：2527 张
* 分类数量：6 类

  * Cardboard（纸板）
  * Glass（玻璃）
  * Metal（金属）
  * Paper（纸张）
  * Plastic（塑料）
  * Trash（其他垃圾）

数据集存在一定类别不均衡，其中 **Paper** 类样本较多，而 **Trash** 类样本相对较少。

数据划分方式如下：

| 数据集        | 比例  |
| ---------- | --- |
| Train      | 70% |
| Validation | 15% |
| Test       | 15% |

采用**分层随机划分（Stratified Split）**，保证各类别比例基本一致。



# 项目结构

```text
.
├── data/                  # 数据集
├── models/                # 网络模型
│   ├── simple_cnn.py
│   ├── advanced_cnn.py
│   ├── resnet18.py
│   └── cbam.py
├── augmentation.py        # 数据增强
├── losses.py              # 损失函数
├── train.py               # 训练
├── evaluate.py            # 测试
├── gradcam.py             # Grad-CAM可视化
├── checkpoints/           # 模型权重
├── results/               # 实验结果
└── README.md
```
# 复现
其中实验二和实验三在kaggle中进行，使用T4GPU
https://www.kaggle.com/code/kageyamafirst/garbage-classification-cnn/edit

# 模型

本项目共比较四种模型配置。

| 模型                    | 简介                            |
| --------------------- | ----------------------------- |
| SimpleCNN             | 三层卷积网络                        |
| AdvancedCNN           | 四层卷积网络，引入 BatchNorm 与 Dropout |
| ResNet-18（Scratch）    | ResNet18 随机初始化训练              |
| ResNet-18（Pretrained） | ImageNet 预训练模型                |


# 数据增强策略

共比较四种增强方案：

| 策略       | 内容                             |
| -------- | ------------------------------ |
| None     | Resize + Normalize             |
| Standard | 随机翻转 + ColorJitter             |
| Weak     | 轻度仿射变换                         |
| Strong   | 大角度旋转 + Random Erasing + 强颜色扰动 |


# 实验一：CNN 与 ResNet 对比

主要探究：

* 网络结构是否影响分类性能？
* ImageNet 预训练是否能够提升模型表现？

## 实验结果

性能排序：

> Pretrained + Standard > Pretrained + None > Scratch + None > Scratch + Standard

最佳配置：

* 模型：ResNet-18（Pretrained）
* 数据增强：Standard

最终性能：

| 指标       | 数值        |
| -------- | --------- |
| Accuracy | **85.1%** |
| Macro-F1 | **0.820** |

### 实验结论

* ImageNet 预训练是影响性能最大的因素。
* Standard 数据增强能够进一步提升预训练模型性能。
* 即使不使用预训练，ResNet18 依然明显优于 SimpleCNN。

---

# 实验二：数据增强策略分析

针对 SimpleCNN 与 AdvancedCNN 比较四种数据增强策略。

## 主要结论

### SimpleCNN

Standard 增强效果最佳。

测试集 Macro-F1：

> 67.23% → **69.47%**

说明数据增强能够有效缓解浅层网络过拟合。



### AdvancedCNN

最佳结果反而来自**无增强（None）**。

说明：模型容量较大时，过强的数据增强可能破坏原有判别特征，导致性能下降。

---

# 实验三：学习率与损失函数

学习率：

* 0.01
* 0.001
* 0.0001
* 0.00001

损失函数：

* CrossEntropyLoss
* Weighted CrossEntropy
* Focal Loss

## 实验结论

### 学习率

学习率对模型性能影响远大于损失函数。

其中：

> **Learning Rate = 0.0001**

获得最佳收敛效果。


### 损失函数

CrossEntropyLoss 获得最高整体性能。

Weighted CrossEntropy：

* 明显提高 Trash 类召回率
* Recall：

  * 0.62 → **0.86**

Focal Loss：

能够降低类别预测偏置，但总体性能仍略低于 CrossEntropyLoss。

---

# 错误分析

混淆矩阵分析发现：

## Glass ↔ Metal

四组实验均持续存在混淆。

原因：

* 外观轮廓相似；
* 表面反光特征接近；
* RGB 特征难以充分区分材质。



## Trash

始终是最难分类类别。

原因：

* 类内差异大；
* 样本数量少；
* 缺乏稳定视觉特征。



# Grad-CAM 可解释性分析

Grad-CAM 可视化结果表明：

正确预测时：

* 模型主要关注垃圾主体轮廓；
* 材质纹理；
* 关键区域。

错误预测时：

模型更容易关注：

* 背景；
* 标签文字；
* 高光反射区域。

说明模型仍存在注意力分散的问题。

---

# 后续改进

引入 **CBAM（Convolutional Block Attention Module）**。

预期目标：

* 提高模型注意力集中能力；
* 抑制背景噪声；
* 改善 Glass 与 Metal 等易混类别的识别效果；
* 提升少数类别（Trash）的分类性能。

# 实验结论

本项目主要得到以下结论：

1. **ImageNet 预训练是影响模型性能最重要的因素。**

2. **数据增强需要与模型容量相匹配。** 对于 SimpleCNN，Standard 增强能够有效提升泛化能力；而对于 AdvancedCNN，无增强反而取得最佳结果。

3. **学习率的重要性高于损失函数。** 在本实验中，学习率 0.0001 获得最佳性能。

4. **CrossEntropyLoss 整体表现最佳。** Weighted CrossEntropy 能够显著提升少数类召回率，但整体精度略有下降。

5. **Glass 与 Metal 的混淆是当前模型最主要的错误来源。** 后续计划结合注意力机制进一步提升模型判别能力。



# 运行环境

* Python 3.11
* PyTorch
* torchvision
* NumPy
* scikit-learn
* Matplotlib

---

# 致谢

感谢 Kaggle 提供 Garbage Classification 数据集，感谢 PyTorch 社区提供开源深度学习框架，为本项目实验提供了支持，感谢《人工智能导论》课程给了本次CV作业提供了坚实的知识储备与答疑，让我们对知识本身有了更深层次的理解！
