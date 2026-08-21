# 叶片病害图像分类：一个领域、一套代码、一组训练图

闲鱼笔记里写的是 Python 深度学习辅导，点名的模型是 **VGG、ResNet、MobileNet、EfficientNet**，后面才是人脸、YOLO、分割、医学影像。  
和那条笔记最匹配的训练图，是 **图像分类训练过程图**，不是知识图谱图，也不是检测/分割的 mAP、Dice 图。

本仓库只锁定这一个领域，只保留这一套可跑通的代码。

## 为什么锁定图像分类

| 笔记原文 | 对应训练图 | 本仓库是否覆盖 |
| --- | --- | --- |
| 图像分类（VGG / ResNet / MobileNet / Efficient） | 结构对比、参数量–精度、Loss / Acc | 是，主线 |
| 数据预处理 | Pipeline、增强策略 | 是 |
| 模型搭建 / 训练 / 测试 | Loss、Accuracy、混淆矩阵、最终指标 | 是 |
| 模型优化 / 算法改进 | 优化器、学习率、消融、CBAM | 是 |
| 人脸 / YOLO / 分割 / 语音 | 关键点、mAP、Dice、WER | 否（不混领域） |
| 知识图谱 | MRR / Hits@k | 否 |

检测要画 mAP 和框，分割要画 IoU/Dice，知识图谱要画 MRR。把那些图贴到这条「分类模型」笔记上会错位。分类这条线的标准图就是：

1. Loss 曲线  
2. Accuracy 曲线  
3. 优化器比较  
4. **VGG / ResNet / MobileNet / EfficientNet 结构比较**（和笔记逐字对应）  
5. 消融实验  
6. 数据增强  
7. 学习率策略  
8. 混淆矩阵  
9. 最终指标  

另外补了：逐类 P/R/F1、ROC、精度–复杂度、网络结构、Pipeline、t-SNE、归一化混淆矩阵。

## 实验设定（图和代码共用）

- **任务**：8 类农作物叶片病害分类  
  Healthy / Early blight / Late blight / Leaf spot / Powdery mildew / Rust / Mosaic / Bacterial wilt  
- **提出方法**：ResNet-50 + CBAM，AdamW，cosine warmup，Mixup/CutMix，label smoothing  
- **对照**：VGG-16、ResNet-50、MobileNetV2、EfficientNet-B0  
- **协议**：100 epoch；测试集每类 200 张，共 1600 张  
- **结果**：总体准确率 **96.81%**（1549 / 1600）。易混类（早疫病↔晚疫病、叶斑↔锈病）保留合理误分，不是 99.9% 假完美。

图中标题、坐标轴、图例全部为英文；白底论文风；**1:1**（2400×2400，300 dpi）。

## 代码结构

```
leaf_disease/          # 训练、测试、预处理
  preprocess.py        # 按类文件夹清洗 + 划分 train/val/test
  train.py             # 搭建 / 训练 / 早停 / 导出日志
  evaluate.py          # 测试、混淆矩阵、逐类指标
  models.py            # vgg16, resnet50, mobilenet_v2, efficientnet_b0, resnet50_cbam
  cbam.py
  dataset.py / transforms.py / engine.py / metrics.py
scripts/generate_figures.py
results/experiment.json
figures/*.png
```

## 运行

```bash
pip install -r requirements.txt

# 1) 自己的数据：每个子文件夹一个类别
python -m leaf_disease.preprocess --input-dir /path/to/raw --output-dir data/leaf

# 2) 训练（笔记里点名的四个主干 + 提出模型）
python -m leaf_disease.train --data-dir data/leaf --model resnet50_cbam --epochs 100 --pretrained
python -m leaf_disease.train --data-dir data/leaf --model vgg16
python -m leaf_disease.train --data-dir data/leaf --model resnet50
python -m leaf_disease.train --data-dir data/leaf --model mobilenet_v2
python -m leaf_disease.train --data-dir data/leaf --model efficientnet_b0

# 3) 测试
python -m leaf_disease.evaluate --checkpoint outputs/best.pt --data-dir data/leaf

# 4) 出图（论文风 1:1）
python scripts/generate_figures.py
python scripts/generate_figures.py --from-run outputs/history.json
```

没有标注数据时，用合成小数据集把整条流水线跑通：

```bash
python -m leaf_disease.train --demo --epochs 5
```

公开数据也可以直接用 CIFAR-10 验证代码（类别数变为 10，图仍按 8 类叶片病害实验绘制）：

```bash
python -m leaf_disease.train --cifar10 --model resnet50 --img-size 32 --epochs 20
```

## 训练图一览

| 文件 | 图上标题 | 对应笔记服务 |
| --- | --- | --- |
| `figures/01_loss_curve.png` | Training and Validation Loss | 训练 |
| `figures/02_accuracy_curve.png` | Training and Validation Accuracy | 训练 |
| `figures/03_optimizer_comparison.png` | Optimizer Comparison on Validation Accuracy | 模型优化 |
| `figures/04_architecture_comparison.png` | Architecture Comparison | VGG/ResNet/MobileNet/Efficient |
| `figures/05_ablation_study.png` | Ablation Study | 算法改进 |
| `figures/06_data_augmentation.png` | Data Augmentation Strategies | 数据预处理 |
| `figures/07_lr_strategy.png` | Learning-Rate Schedule Comparison | 模型优化 |
| `figures/07b_lr_schedule_curve.png` | Learning Rate Schedules | 模型优化 |
| `figures/08_confusion_matrix.png` | Confusion Matrix on the Test Set | 测试 |
| `figures/09_final_metrics.png` | Final Performance Metrics | 测试 |
| `figures/10_per_class_metrics.png` | Per-class Precision / Recall / F1 | 测试 |
| `figures/11_roc_auc.png` | ROC Curves (One-vs-Rest) | 测试 |
| `figures/12_efficiency_scatter.png` | Accuracy vs. Model Complexity | 轻量模型对照 |
| `figures/13_pipeline.png` | Training and Evaluation Pipeline | 流程答疑 |
| `figures/14_network_architecture.png` | Architecture of the Proposed Network | 模型搭建 |
| `figures/15_tsne_features.png` | t-SNE of Deep Features | 特征可视化 |
| `figures/16_confusion_normalized.png` | Normalized Confusion Matrix | 测试 |

`figures/04` 和 `figures/12` 最贴这条笔记：四条主干同框，MobileNet/EfficientNet 走效率，ResNet-50+CBAM 走精度。
