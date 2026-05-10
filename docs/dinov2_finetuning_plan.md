# DINOv2 Fine-Tuning Plan

## Objective

Train a lightweight classifier head on top of frozen DINOv2-base patch features
to predict geometry-derived pothole severity directly from visual embeddings.

## Why DINOv2?

DINOv2 (Oquab et al., 2024) is a self-supervised vision foundation model trained
on 142M images. Its patch-level features encode rich material, texture, and
structural information without any task-specific fine-tuning.

**Key advantage**: DINOv2 features generalize across domains, camera types, and
lighting conditions far better than hand-crafted features. This is critical for
pothole detection where training data is limited and deployment conditions are diverse.

## Architecture

```
Input Image (224×224)
    │
    ▼
DINOv2-base (FROZEN)
    │
    ├── [CLS] token (768-dim)  ← global image representation
    │
    ▼
Linear(768 → 256) + ReLU + Dropout(0.3)
    │
    ▼
Linear(256 → 3) + Softmax
    │
    ▼
[Shallow | Moderate | Deep]
```

**Total trainable parameters**: ~200K (only the classifier head)
**Frozen parameters**: 86M (DINOv2-base backbone)

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for transformer fine-tuning |
| Learning rate | 1e-4 | Conservative — only training head |
| Weight decay | 0.01 | Light regularization |
| Batch size | 32 | Fits in 8GB GPU memory with frozen backbone |
| Epochs | 20 | Small dataset, fast convergence expected |
| LR schedule | Cosine annealing | Smooth decay prevents oscillation |
| Loss | Cross-entropy with class weights | Handles class imbalance (fewer Deep samples) |

## Class Weights

Compute from training set distribution:
```
weight_i = total_samples / (num_classes × count_i)
```

Expected approximate weights:
- Shallow: 0.8 (most common)
- Moderate: 1.0 (baseline)
- Deep: 1.5 (least common, most important)

## Training Data

- **Source**: merged_dataset with geometry-derived severity labels
- **Split**: Same train/valid split as ml_classifier.py (ensures fair comparison)
- **Labels**: Pseudo-labels from `generate_pseudo_labels()` in ml_classifier.py
- **Augmentation**: RandomHorizontalFlip, RandomRotation(±15°), ColorJitter(0.1)

## Evaluation Plan

Compare DINOv2 classifier against tabular feature classifiers on the same
validation split:

| Model | Input | Expected Accuracy |
|-------|-------|-------------------|
| Random Forest (11 features) | Depth statistics | Baseline |
| Random Forest (20 features) | Extended features | +2-5% |
| Random Forest (geometry) | Curvature + depth profile | +3-7% |
| **DINOv2 classifier** | **Raw image patches** | **+5-10%** |

**Key hypothesis**: DINOv2 should outperform tabular classifiers because it can
learn complex visual patterns (material degradation, crack networks, water
presence) that are difficult to capture with hand-crafted features.

## Implementation Steps

1. Create `scripts/train_dinov2_classifier.py`
2. Use `transformers` + `torch` for model and training loop
3. Save trained head to `ml_models/dinov2_classifier_head.pt`
4. Add prediction function to `foundation_features.py`
5. Integrate into inference pipeline (optional, behind flag)

## Hardware Requirements

- **Minimum**: CPU-only training feasible (frozen backbone, small head)
- **Recommended**: Single GPU with 8GB VRAM (3-5 min/epoch)
- **Inference**: ~100ms per image on CPU, ~15ms on GPU
