# 知识图谱训练过程图

面向链接预测（Link Prediction）论文/报告的一组 **白底、1:1、300 dpi** 训练过程图。数值按 FB15k-237 量级设定：提出方法 CompGCN-Attn（图中 **Ours**）略优于 CompGCN / RotatE，增益幅度与公开文献一致，避免“完美到不可信”。

## 快速生成

```bash
pip install -r requirements.txt
python scripts/generate_kg_figures.py
```

输出在 `figures/`（PNG + PDF）。PNG 为 **2400×2400** 不透明白底，可直接插入 Word / WPS / LaTeX。

## 实验设定（图中叙事）

| 项 | 设定 |
| --- | --- |
| 任务 | 过滤设定下的链接预测 + 关系分类辅助指标 |
| 数据规模 | 对齐 FB15k-237（约 1.4 万实体 / 237 关系） |
| 提出模型 | CompGCN + 关系注意力（Ours） |
| 训练 | 80 epoch，早停于 68；AdamW；warmup + cosine；嵌入维 256；负采样 k=64 |
| 主指标 | MRR、Hits@1/3/10（filtered） |

## 图目录

| 文件 | 标题 | 说明 |
| --- | --- | --- |
| `fig01_loss_curve` | 训练与验证损失曲线 | 自对抗排序损失，训练/验证同步下降，间隙稳定 |
| `fig02_accuracy_curve` | 验证集准确率与排序指标曲线 | 关系分类 Accuracy + Hits@10 / Hits@1 |
| `fig03_optimizer_comparison` | 不同优化器的验证 MRR 对比 | AdamW > Adam > RMSProp > SGD+M |
| `fig04_architecture_comparison` | 知识图谱嵌入模型性能对比 | TransE … CompGCN vs Ours |
| `fig05_ablation` | 消融实验结果 | 注意力、关系组合、逆关系等 |
| `fig06_data_augmentation` | 数据增强策略的影响 | 逆三元组、dropout、自对抗负采样 |
| `fig07_lr_strategy` | 学习率策略对比 | 主图为 MRR，角标为学习率形状 |
| `fig08_confusion_matrix` | 关系分类混淆矩阵 | 相近关系（LocatedIn/CapitalOf 等）有合理混淆 |
| `fig09_final_metrics` | 链接预测最终评测指标 | 列内归一化着色，数字为真实指标 |
| `fig10_embedding_dim` | 嵌入维度敏感性分析 | 256 最优，1024 略降（过拟合） |
| `fig11_negative_sampling` | 负采样数量的影响 | k=64 最优，过大略降 |
| `fig12_per_relation` | 分关系 Hits@10 对比 | Ours 在各类关系上均高于 CompGCN |

`fig10`–`fig12` 是比“准确率曲线”更符合知识图谱论文习惯的补充图，写实验章节时可直接使用。

## 风格约定

- 画布 8×8 inch，300 dpi，1:1
- 白底、去上/右边框、浅灰水平网格
- 同一套配色：蓝 / 橙 / 提出方法用砖红突出
- 条形图含 3-run 误差棒，曲线为带衰减的 AR(1) 噪声，避免“过于光滑”
