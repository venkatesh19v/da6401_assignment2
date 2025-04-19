# Fine-Tuning Pretrained Models on iNaturalist

This project focuses on **fine-tuning ImageNet-pretrained models**—such as **Vision Transformer (ViT-B/16), EfficientNetV2, ResNet, VGG**, etc.—on a 10-class subset of the **iNaturalist** dataset. Instead of training from scratch, we leverage transfer learning to achieve significantly higher performance with fewer resources and training time.

---

## Project Structure

```
.
├── FineTuneModel.py                   # PyTorch Lightning module for fine-tuning
├── train.py                           # Script to train the model
├── test.py                            # Script to evaluate the fine-tuned model
├── sweep.py                           # W&B sweep script for hyperparameter tuning
└── random_test_predictions_grid.png  # Visual output of model predictions
```

---

## Models & Strategies Used

Explored multiple pre-trained backbones and fine-tuning strategies:

### Models:
- Vision Transformer (ViT-B/16)
- EfficientNet V2-S
- VGG16
- ResNet18, ResNet50
- AlexNet

### Fine-Tuning Strategies:
- **Freeze All** – Only train the final classification layer.
- **Freeze Until Layer4** – Freeze early layers, train last few layers + classifier.
- **Unfreeze All** – Train the full model end-to-end.

---

## Key Results

| Model             | Freeze All | Freeze Until Layer4 | Unfreeze All |
|------------------|------------|----------------------|---------------|
| **ViT B/16**      | 83.50%     | **84.50%**           | 24.60%        |
| EfficientNet V2-S | 72.80%     | 84.20%               | 70.70%        |
| VGG16             | 75.40%     | 77.00%               | 28.40%        |
| ResNet50          | 75.40%     | 72.50%               | 51.10%        |
| ResNet18          | 67.70%     | 70.50%               | 54.50%        |
| AlexNet           | 63.00%     | 62.10%               | 10.00%        |

> **Best Result**: ViT-B/16 + Freeze Until Layer4 — **84.50% Test Accuracy**

---

## Insights:

### Fine-Tuning vs Training from Scratch

- **Pretrained ViT-B/16**:  
  Achieved **84.50% test accuracy** with just 10 epochs using partial freezing.
  
- **Simple CNN (from Part A)**:  
  Only reached ~35–40% test accuracy with much longer training.

### Strategic Freezing Matters

- **freeze_until_layer4** struck the best balance between flexibility and efficiency.
- **Unfreezing all** led to overfitting or unstable learning on small datasets.
- **Freezing all** was efficient, but underperformed compared to partial tuning.

---

## Training Configuration

| Setting         | Value              |
|----------------|--------------------|
| Batch Size      | 32                 |
| Learning Rate   | 1e-3               |
| Epochs          | 10                 |
| Optimizer       | Adam               |
| Loss Function   | CrossEntropyLoss   |
| Logger          | W&B (Weights & Biases) |
| Checkpointing   | On `val_accuracy`  |
| Data Split      | 80% train / 20% val (stratified) |
| Test Set        | Held-out set       |

---

### Run a W&B Sweep

To identify which model and fine-tuning strategy yields the best performance:

```bash
python sweep.py
```
---

### Train the Model

Train the model using the **best-performing configuration**:

- **Model**: Vision Transformer (ViT-B/16)  
- **Strategy**: `freeze_until_layer4`  
- **Test Accuracy**: **84.50%**

```bash
python train.py
```
---

### Evaluate on the Test Set

- Evaluate the fine-tuned model on the held-out test dataset.  
- Predictions from the model are visualized in `random_test_predictions_grid.png`.

```bash
python test.py
```
## Prediction on the Test dataset:

### Model trained from Finetuned model:  ViT-B/16 (`freeze_until_layer4`)
![Prediction Grid](/partB/random_test_predictions_grid.png)
---
