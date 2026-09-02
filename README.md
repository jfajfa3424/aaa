# 目标检测训练图 + 检测效果图

Urban traffic object detection figure pack (YOLOv8s + BiFPN + CBAM).
On-figure text is English. 8 classes: person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign.

Regenerate:

```bash
pip install -r requirements.txt
python scripts/generate_figures.py
```

Detection boxes in `figures/det_*.png` are real YOLOv8s predictions on the six demo scenes (`assets/detections.json`). Training/ablation curves are a self-consistent paper-style experiment (mAP@0.5 = 92.47%).

## 闲鱼笔记标题（可直接用）

主推：

**YOLOv8目标检测实战笔记｜训练曲线+检测框效果图全套可发**

备选：

1. 目标检测从 Loss 到 mAP 全过程｜8 类交通场景可视化
2. 一份能发笔记的目标检测出图包｜曲线 · 混淆矩阵 · 实拍检测框
3. YOLOv8 交通目标检测｜白天路口 / 夜间雨天 / 高速 / 停车场效果对比
4. 目标检测训练可视化合集｜Box/Cls/DFL 损失 + PR 曲线 + F1-Confidence
5. 改进 YOLOv8（BiFPN+CBAM）｜消融实验 + 速度精度对比 + 检测效果

封面图建议用 `figures/23_detection_mosaic.png`。训练图发笔记优先用拼图，不必一张张传：

- `C3_results_2x5.png` YOLO 风格 2×5 训练仪表盘  
- `C4_training_3x3.png` 九宫格训练板  
- `C5_eval_2x3.png` 评估六宫格

## 图片清单

### 训练 / 指标（1:1，适合论文和笔记正文）

| 文件 | 内容 |
| --- | --- |
| `01_loss_curve.png` | Box / Cls / DFL 训练与验证损失 |
| `02_map_curve.png` | mAP@0.5 与 mAP@0.5:0.95 |
| `03_precision_recall.png` | Precision / Recall 随 epoch |
| `04_optimizer_comparison.png` | SGD / Adam / AdamW |
| `05_architecture_comparison.png` | YOLOv5s / v7-tiny / v8s / RT-DETR / Ours |
| `06_ablation_study.png` | BiFPN、CBAM、Mosaic、CIoU+NWD |
| `07_data_augmentation.png` | 数据增强策略 |
| `08_lr_strategy.png` | 学习率策略对比 |
| `08b_lr_schedule_curve.png` | 学习率曲线 |
| `09_confusion_matrix.png` | 混淆矩阵 |
| `10_final_metrics.png` | P / R / mAP 总表 |
| `11_per_class_ap.png` | 各类 AP |
| `12_pr_curve.png` | PR 曲线 |
| `13_f1_confidence.png` | F1–Confidence（YOLO 风格） |
| `14_radar_metrics.png` | 雷达图 |
| `15_efficiency_scatter.png` | FPS vs mAP |
| `16_confusion_normalized.png` | 归一化混淆矩阵 |
| `17_hyperparam_heatmap.png` | 超参热力图 |
| `18_seed_std_band.png` | 多种子标准差带 |
| `19_network_architecture.png` | 网络结构 |
| `20_pipeline.png` | 训练流水线 |
| `21_robustness.png` | 噪声鲁棒性 |
| `22_label_wh.png` | 框宽高分布 |
| `24_detection_grid.png` | 2×2 检测效果 |
| `25_before_after.png` | 原图 vs 检测 |
| `26_class_legend.png` | 类别配色 |
| `C1_training_2x2.png` | 论文拼图：训练 |
| `C2_evaluation_2x2.png` | 论文拼图：评估 |

### 检测效果（16:9，适合封面和轮播）

| 文件 | 场景 |
| --- | --- |
| `23_detection_mosaic.png` | 六宫格封面 |
| `det_crosswalk.png` | 斑马线路口 |
| `det_street.png` | 城市街道 |
| `det_highway.png` | 高速公路 |
| `det_parking.png` | 停车场 |
| `det_night.png` | 夜间雨天 |
| `det_corner.png` | 街角停车标志 |
