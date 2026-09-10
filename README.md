# 闲鱼笔记用机器学习结果图

一套暗色科技风的训练结果展示图，适合做闲鱼笔记**头图**和**详情图**。

指标是演示用样例（R² 0.968、AUC 0.971、MAPE 3.8%），看起来正向、专业，但不要写成某个真实客户的交付结论。

## 怎么用

| 文件 | 建议用途 |
| --- | --- |
| `output/00_cover_square.png` | 头图（1:1，信息密度高） |
| `output/01_cover_portrait.png` | 头图备选（3:4，手机信息流更醒目） |
| `output/02_loss_accuracy.png` | 详情：训练过程 |
| `output/03_model_compare.png` | 详情：多模型对比 |
| `output/04_pred_vs_actual.png` | 详情：回归拟合 |
| `output/05_timeseries_lstm.png` | 详情：LSTM 时序 |
| `output/06_feature_importance.png` | 详情：可解释性 |
| `output/07_classification_roc.png` | 详情：分类效果 |
| `output/08_kmeans_clusters.png` | 详情：用户分群 |

建议顺序：`01` 或 `00` 做封面，详情按 `02 → 03 → 04 → 05 → 07 → 06 → 08`。

## 重新出图

```bash
pip install -r requirements.txt
python generate_xianyu_charts.py
```

需要系统中文字体（脚本默认 `WenQuanYi Micro Hei`）。
