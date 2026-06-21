# YOLOv8n Cheque Detection — Augmentation Ablation Study

A controlled ablation study measuring how data augmentation intensity affects YOLOv8n's ability to detect cheques under severe data scarcity (177 images). Four training configurations — minimal, light, heavy, and a proposed balanced augmentation recipe — are compared head-to-head under otherwise identical training conditions.

📄 Full writeup: [`paper/cheque_augmentation_ablation.pdf`](paper/cheque_augmentation_ablation.pdf) (IEEE conference format)

## Research Question

> Under severe data scarcity, does aggressive augmentation help or hurt a structured-document detector — and where's the sweet spot?

Augmentation recipes shipped with most detection frameworks (mosaic, copy-paste, mixup, large-angle rotation) were tuned on natural-image datasets with thousands of examples per class. This study tests whether that intensity transfers to a rigid-layout, single-class, ~170-image document detection task.

## Results

| Variant | mAP@50 | mAP@50-95 | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|
| A — Baseline | 0.9950 | 0.6077 | 0.9820 | 1.0000 | 48 |
| B — Aug-Light | 0.9791 | 0.5827 | 0.9136 | 0.9444 | 23 |
| C — Aug-Heavy | 0.9945 | 0.4808 | 0.9866 | 0.9722 | 47 |
| **D — Aug-Balanced** | **0.9950** | **0.6087** | **0.9977** | **1.0000** | 46 |

**Takeaway:** the proposed balanced configuration (moderate mosaic, layout-preserving geometry, light copy-paste, cosine LR) wins on every metric and Pareto-dominates the precision/recall trade-off. Heavy augmentation shows the smallest train/validation generalization gap but the worst mAP@50-95 — augmentation intensity trades optimization difficulty against overfitting resistance even at this small scale. Full discussion in the paper.

![Best-epoch metric comparison](figures/fig1_metric_comparison_bar.png)

## Repository Structure

```
.
├── paper/
│   ├── cheque_augmentation_ablation.tex   # IEEE conference-format source
│   └── cheque_augmentation_ablation.pdf   # compiled paper
├── figures/
│   ├── fig1_metric_comparison_bar.png
│   ├── fig2_map50_curves.png
│   ├── fig3_map50_95_curves.png
│   ├── fig4_train_loss_curves.png
│   ├── fig5_generalization_gap.png
│   └── fig6_precision_recall_scatter.png
├── models/
│   ├── variant_A_baseline/
│   │   ├── best.pt
│   │   └── best_float16.tflite
│   ├── variant_B_aug_light/
│   │   ├── best.pt
│   │   └── best_float16.tflite
│   ├── variant_C_aug_heavy/
│   │   ├── best.pt
│   │   └── best_float16.tflite
│   └── variant_D_aug_balanced/
│       ├── best.pt
│       └── best_float16.tflite
├── results/
│   └── ablation_results.json              # full per-epoch training curves + metrics
└── README.md
```

## Models

Each variant's trained weights are provided in two formats:

- **`best.pt`** — native Ultralytics/PyTorch checkpoint
- **`best_float16.tflite`** — float16-quantized TFLite export, for mobile/on-device inference

`variant_D_aug_balanced` is the recommended model for production use based on this study's results.

### Inference with PyTorch (`best.pt`)

```python
from ultralytics import YOLO

model = YOLO("models/variant_D_aug_balanced/best.pt")
results = model.predict("path/to/cheque_image.jpg", conf=0.5)
results[0].show()
```

### Inference with TFLite (`best_float16.tflite`)

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="models/variant_D_aug_balanced/best_float16.tflite")
interpreter.allocate_tensors()
# Preprocess image to 640x640, run interpreter.invoke(), decode YOLO output as usual
```

## Methodology Summary

All four variants share: YOLOv8n backbone, 120-epoch budget, 640×640 input, batch size 16, seed 42, 80/20 train/val split (141/36 images). Only augmentation parameters and their co-tuned optimization settings (learning rate, weight decay, patience, LR schedule) differ between variants. Full hyperparameter tables and design rationale per variant are in the paper (Section IV).

| | A: Baseline | B: Aug-Light | C: Aug-Heavy | D: Aug-Balanced |
|---|---|---|---|---|
| Mosaic | 0.3 | 0.1 | 1.0 | 0.5 |
| Rotation | 0° | 0° | 15° | 5° |
| Shear / Perspective | — | — | ✓ | — |
| Copy-paste | — | — | 0.2 | 0.1 |
| Mixup | — | — | 0.1 | — |
| LR schedule | constant | constant | constant | cosine |

## Dataset

177 annotated cheque images, single class (`cheque`). The raw dataset is **not included** in this repository due to the sensitivity of financial document images. Reach out if you have a legitimate research use case.

## Raw Results

Full per-epoch training/validation curves, hyperparameters, and best-epoch metrics for all four variants are available in [`results/ablation_results.json`](results/ablation_results.json) for anyone who wants to reproduce the figures or run further analysis.

## Citation

If you use this work, please cite:

```bibtex
@unpublished{cheque_aug_ablation,
  title  = {Augmentation Intensity vs. Generalization in YOLOv8n for Cheque Detection Under Severe Data Scarcity},
  author = {Your Name},
  year   = {2026},
  note   = {Independent research}
}
```

## License

Specify your license here (e.g., MIT, Apache 2.0, or "All rights reserved" if you don't want to license model weights/code for reuse yet).

## Author

Your Name — Independent Researcher