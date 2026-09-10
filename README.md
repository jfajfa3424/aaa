# 医学影像组学 / 病理组学训练结果图包

面向闲鱼笔记的论文风格出图包：CT/MRI/PET 分割、Swin-UNETR、多模态融合、传统+深度影像组学、预后列线图。

图上科学标注为英文（和期刊一致）。封面图带中文标题，方便直接当头图。

**下载：** 仓库根目录 `01-头图-…png` 到 `20-详情-…png` 可直接保存；全部打包见 `医学影像组学出图包.zip`。完整原图仍在 `figures/`。

一套自洽的示意实验（NSCLC，CECT + 18F-FDG PET + 临床）：

| 指标 | 数值 |
| --- | --- |
| GTV Dice (CT / PET) | 0.914 / 0.889 |
| IoU / HD95 | 0.841 / 3.82 mm |
| 亚型 4 分类准确率 | 94.06% |
| 3 年 OS AUC（训练 / 验证 / 测试 / 外部） | 0.921 / 0.894 / 0.887 / 0.862 |
| Nomogram C-index（训练 / 测试） | 0.821 / 0.786 |
| vs TNM 的 ΔAUC | +0.193 |

重新生成：

```bash
pip install -r requirements.txt
python scripts/generate_figures.py
```

## 闲鱼笔记标题（可直接用）

主推：

**医学影像组学实战笔记｜Swin分割+ROC/DCA/列线图全套可发**

备选：

1. 影像组学从勾画到预后｜Dice 0.91 + 列线图 + KM 曲线
2. CT/MRI/PET 多模态融合出图包｜分割叠加 · CAM · 病理组学
3. Swin-UNETR 医学分割训练板｜Loss/Dice/HD95 一篇发完
4. 临床科研标配四件套｜ROC · DCA · 校准 · 列线图
5. 一对一指导：病灶分割标注、组学特征、预后预测怎么出图

## 头图怎么选（按点击率）

发笔记时**第一张必须是封面**，不要丢单张折线图：

| 优先级 | 文件 | 用途 |
| --- | --- | --- |
| 1 | `COVER_01_seg_mosaic.png` | **首选头图**：九宫格分割叠加，绿=金标准、红=预测 |
| 2 | `COVER_03_multimodal.png` | 多模态：CT / PET / 融合 / MRI / 病理 |
| 3 | `COVER_02_clinical.png` | 临床科研感：ROC + KM + DCA + 列线图 |
| 4 | `COVER_05_figure1.png` | 论文 Figure 1 排版，适合当第二张详情 |
| 5 | `COVER_04_banner_16x9.png` | 16:9 横版，适合店铺横幅 |

详情轮播建议顺序：

1. `COVER_01_seg_mosaic.png`
2. `COVER_05_figure1.png`
3. `C5_results_2x5.png`（训练仪表盘）
4. `C2_clinical_2x2.png`
5. `21_seg_overlay.png` 或 `22_seg_strip.png`
6. `17_nomogram.png`
7. `15_dca.png` + `18_kaplan_meier.png`
8. `24_attention_cam.png` + `23_multimodal.png`

## 图片清单

### 训练过程（1:1，适合正文）

| 文件 | 内容 |
| --- | --- |
| `01_loss_curve.png` | Dice+CE 训练/验证损失 |
| `02_dice_curve.png` | Dice 曲线，最佳 0.914 |
| `03_hd95_iou.png` | HD95 与 IoU 双轴 |
| `04_optimizer_comparison.png` | SGD / Adam / AdamW |
| `05_architecture_comparison.png` | U-Net → Swin-UNETR |
| `06_ablation_study.png` | 注意力 / 深监督 / PET 融合 / 边界损失 |
| `07_lr_schedule.png` | cosine + warmup |
| `08_confusion_matrix.png` | ADC/SCC/LCC/SCLC |
| `09_final_metrics.png` | 分割 + 分型 + 预后总表 |
| `10_per_class_dice.png` | GTV 与危及器官 |
| `11_seed_std_band.png` | 三随机种子标准差带 |
| `35_hyperparam_heatmap.png` | 学习率 × 权重衰减 |

### 临床统计（发影像组学论文最常用）

| 文件 | 内容 |
| --- | --- |
| `12_roc_auc.png` | TNM / 临床 / 组学 / 列线图 ROC |
| `13_roc_splits.png` | 训练/验证/测试/外部 |
| `14_pr_curve.png` | 亚型 PR |
| `15_dca.png` | 决策曲线（临床净获益） |
| `16_calibration.png` | 3 年 OS 校准 |
| `17_nomogram.png` | 预后列线图 |
| `18_kaplan_meier.png` | 高低危 KM，p < 0.001 |
| `19_forest_plot.png` | 多因素 Cox 森林图 |
| `20_radar_metrics.png` | 多指标雷达图 |

### 可视化（好看、好懂、好成交）

| 文件 | 内容 |
| --- | --- |
| `21_seg_overlay.png` | 四例 GTV 叠加 |
| `22_seg_strip.png` | CT / GT / Pred / Overlay |
| `23_multimodal.png` | CT · PET · 融合 · 病理 |
| `24_attention_cam.png` | Swin Grad-CAM |
| `25_radiomics_heatmap.png` | 组学特征热图（高低危） |
| `26_tsne_features.png` | 深度特征 t-SNE |
| `27_swin_architecture.png` | Swin-UNETR 结构 |
| `28_pipeline.png` | 预处理→ROI→特征→模型 |
| `29_video_swin.png` | Video-Swin 纵向 MRI 疗效 |
| `30_gtv_3d.png` | 三维病灶表面 |
| `31_feature_importance.png` | SHAP / LASSO 重要性 |
| `32_radscore_boxplot.png` | 亚型 rad-score |
| `33_waterfall.png` | 体积变化瀑布图 |
| `34_preprocess_roi.png` | 窗宽窗位 / ROI 裁剪 |

### 拼图（笔记详情优先发这些，不必一张张传）

| 文件 | 内容 |
| --- | --- |
| `C1_training_2x2.png` | 训练四宫格 |
| `C2_clinical_2x2.png` | ROC · DCA · KM · 列线图 |
| `C3_training_3x3.png` | 九宫格训练板 |
| `C4_eval_2x3.png` | 评估六宫格 |
| `C5_results_2x5.png` | YOLO 风格 2×5 训练仪表盘 |
