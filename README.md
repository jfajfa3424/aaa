# Image classification training figures

Paper-style figure set for an 8-class leaf-disease classifier. Every figure is white-background, 1:1 (2400×2400, 300 dpi), with an **English title** on the image.

## Experiment story

- **Task**: 8-class leaf disease classification (Healthy / Early blight / Late blight / Leaf spot / Powdery mildew / Rust / Mosaic / Bacterial wilt)
- **Proposed method**: ResNet-50 + CBAM, AdamW, cosine annealing with warmup, Mixup/CutMix, label smoothing
- **Protocol**: 100 epochs; 200 test images per class (1600 total)
- **Headline result**: **96.81%** overall accuracy (1549 / 1600), consistent across loss, accuracy, confusion matrix, and metric tables

Confusions are concentrated among visually similar classes (early ↔ late blight, leaf spot ↔ rust), not a perfect diagonal.

## Figure list

### Core training set

| File | Title on figure |
| --- | --- |
| `figures/01_loss_curve.png` | Training and Validation Loss |
| `figures/02_accuracy_curve.png` | Training and Validation Accuracy |
| `figures/03_optimizer_comparison.png` | Optimizer Comparison on Validation Accuracy |
| `figures/04_architecture_comparison.png` | Architecture Comparison |
| `figures/05_ablation_study.png` | Ablation Study |
| `figures/06_data_augmentation.png` | Data Augmentation Strategies |
| `figures/07_lr_strategy.png` | Learning-Rate Schedule Comparison |
| `figures/07b_lr_schedule_curve.png` | Learning Rate Schedules |
| `figures/08_confusion_matrix.png` | Confusion Matrix on the Test Set |
| `figures/09_final_metrics.png` | Final Performance Metrics |
| `figures/10_per_class_metrics.png` | Per-class Precision / Recall / F1 |

### Advanced set (thesis / review)

| File | Title on figure |
| --- | --- |
| `figures/11_roc_auc.png` | ROC Curves (One-vs-Rest) |
| `figures/12_pr_curve.png` | Precision–Recall Curves |
| `figures/13_tsne_features.png` | t-SNE of Deep Features |
| `figures/14_radar_metrics.png` | Multi-Metric Radar Comparison |
| `figures/15_kfold_cv.png` | Five-Fold Cross-Validation |
| `figures/16_efficiency_scatter.png` | Accuracy vs. Model Complexity |
| `figures/17_confusion_normalized.png` | Normalized Confusion Matrix |
| `figures/18_hyperparam_heatmap.png` | Hyperparameter Sensitivity |
| `figures/19_seed_std_band.png` | Validation Accuracy with 3-Seed Std. |
| `figures/20_network_architecture.png` | Architecture of the Proposed Network |
| `figures/21_gradcam.png` | Grad-CAM Visualization |
| `figures/22_robustness.png` | Robustness to Gaussian Noise |
| `figures/23_pipeline.png` | Training and Evaluation Pipeline |

## Regenerate

```bash
pip install -r requirements.txt
python generate_figures.py
```
