# 图像分类训练过程图

面向植物叶片病害 8 分类实验的一套论文风格训练图。全部为白底、1:1（2400×2400，300 dpi），图内含标题，指标互相自洽。

## 实验设定（图中叙事）

- **任务**：8 类叶片病害图像分类（健康 / 早疫病 / 晚疫病 / 叶斑病 / 白粉病 / 锈病 / 花叶病 / 细菌萎蔫）
- **本文方法**：ResNet-50 + CBAM，AdamW，余弦退火 + Warmup，Mixup/CutMix，Label Smoothing
- **训练**：100 epoch；测试集每类 200 张，共 1600 张
- **最终结果**：总体准确率 **96.81%**（混淆矩阵对角和 1549 / 1600），与 Loss / Accuracy / 消融 / 最终指标一致

相近类别会有合理误判（早疫↔晚疫、叶斑↔锈病、白粉↔花叶），而不是接近 100% 的完美对角阵。

## 图件列表

| 文件 | 内容 |
| --- | --- |
| `figures/01_loss_curve.png` | 训练 / 验证损失 |
| `figures/02_accuracy_curve.png` | 训练 / 验证准确率 |
| `figures/03_optimizer_comparison.png` | SGD / RMSprop / Adam / AdamW |
| `figures/04_architecture_comparison.png` | 网络结构：准确率、F1、参数量 |
| `figures/05_ablation_study.png` | 消融：CBAM → Mixup → Cosine Warmup → Label Smoothing |
| `figures/06_data_augmentation.png` | 数据增强逐步叠加 |
| `figures/07_lr_strategy.png` | 学习率策略的验证准确率 |
| `figures/07b_lr_schedule_curve.png` | 学习率数值曲线（含 Warmup 区间） |
| `figures/08_confusion_matrix.png` | 测试集混淆矩阵 |
| `figures/09_final_metrics.png` | Accuracy / Precision / Recall / F1 |
| `figures/10_per_class_metrics.png` | 各类别 Precision / Recall / F1 |

相对常见的 9 张结果图，额外增加了学习率调度曲线和逐类指标，便于和混淆矩阵对照。

## 重新生成

```bash
pip install -r requirements.txt
python generate_figures.py
```
