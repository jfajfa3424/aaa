# Knowledge graph training figures

Paper-style **white-background, 1:1, 300 dpi** figures for a link-prediction study. Numbers follow an FB15k-237-scale setup: the proposed CompGCN-Attn model (**Ours**) slightly outperforms CompGCN / RotatE, with a realistic margin.

Figure titles are in English.

## Generate

```bash
pip install -r requirements.txt
python scripts/generate_kg_figures.py
```

Outputs go to `figures/` (PNG + PDF). PNGs are **2400×2400** opaque white and can be inserted into Word / WPS / LaTeX.

## Experimental setup (narrative used in the figures)

| Item | Setting |
| --- | --- |
| Task | Filtered link prediction + relation classification |
| Data scale | FB15k-237-like (~14.5k entities / 237 relations) |
| Proposed model | CompGCN + relation attention (Ours) |
| Training | 80 epochs, early stop at 68; AdamW; warmup + cosine; dim 256; k=64 negatives |
| Main metrics | MRR, Hits@1/3/10 (filtered) |

## Figure index

| File | Title | Notes |
| --- | --- | --- |
| `fig01_loss_curve` | Training and Validation Loss | Self-adversarial ranking loss; stable train/val gap |
| `fig02_accuracy_curve` | Validation Accuracy and Ranking Metrics | Relation-class accuracy + Hits@10 / Hits@1 |
| `fig03_optimizer_comparison` | Optimizer Comparison on Validation MRR | AdamW > Adam > RMSProp > SGD+M |
| `fig04_architecture_comparison` | Knowledge Graph Embedding Comparison | TransE … CompGCN vs Ours |
| `fig05_ablation` | Ablation Study | Attention, relation composition, inverse relations |
| `fig06_data_augmentation` | Effect of Data Augmentation | Inverse triples, dropout, self-adversarial negatives |
| `fig07_lr_strategy` | Learning-Rate Strategy Comparison | MRR curves with LR-schedule inset |
| `fig08_confusion_matrix` | Relation Classification Confusion Matrix | Plausible mix-ups (LocatedIn/CapitalOf, etc.) |
| `fig09_final_metrics` | Final Link-Prediction Metrics | Column-normalized color; cells show true scores |
| `fig10_embedding_dim` | Embedding Dimension Sensitivity | Best at 256; slight drop at 1024 |
| `fig11_negative_sampling` | Negative Sampling Size | Best at k=64 |
| `fig12_per_relation` | Per-Relation Hits@10 | Ours above CompGCN on every relation |
| `fig13_kg_subgraph` | Predicted Entity–Relation Subgraph | Local typed subgraph; edge width = Hits@10 |

`fig10`–`fig12` are extra figures that fit a typical KG paper better than another generic accuracy plot. `fig13` shows the graph structure itself (entities and relations), using the same relation set and Hits@10 as the training results.

## Style

- 8×8 inch canvas, 300 dpi, 1:1
- White background, no top/right spines, light gray y-grid
- Shared palette: blue / orange, proposed method in brick red
- Bar charts include 3-run error bars; curves use decaying AR(1) noise
