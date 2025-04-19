# CNN Image Classification Project

This repository contains a convolutional neural network (CNN) project implemented using **PyTorch Lightning** and tracked with **Weights & Biases (wandb)**. It includes training scripts, model checkpoints, prediction outputs, and experiment tracking.

---

## Project Structure

```
.
├── best-cnn.ckpt               # Best saved model checkpoint
├── CNN.py                      # CNN model architecture or module definition
├── epoch=9-step=1250.ckpt      # Auto-saved checkpoint from training
├── prediction_grid.png         # Visualization of predictions on a sample grid 10x3
├── Q1.py                       # Script for Question 1 
├── Q2.py                       # Script for Question 2 
├── Q4.py                       # Script for Question 4
├── Question1.ipynb             # Jupyter Notebook version for Question 1
├── README.md                   # Project README (you’re here!)
└── wandb/                      # Weights & Biases run history and metadata
```
---
## Q1:
- 5 Convolution Layers: Each followed by ReLU activation and max-pooling.
- 1 Dense Layer: Fully connected layer before the output layer.
- Output Layer: 10 neurons (corresponding to the 10 classes in the iNaturalist dataset).

```bash
python Q1.py
```
---
## Q2:
**Sweep Config**
```json
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "num_filters":    {"values": [32, 64, 128]},
        "kernel_size":    {"values": [3, 5]},
        "activation":     {"values": ["relu", "gelu", "silu", "mish"]},
        "filter_scheme":  {"values": ["same", "double", "half"]},
        "batch_norm":     {"values": ["True", "False"]},
        "dropout_prob":   {"values": [0.0, 0.2, 0.3]},
        "dense_neurons":  {"values": [64, 128, 256]},
        "lr":             {"min": 1e-4, "max": 1e-2},
        "batch_size":     {"values": [32]},
        "img_size":       {"value": 224},
        "num_classes":    {"value": 10}
    }
```
**Best Sweep Config for Scratch:**
```json
"num_filters": 128,
"kernel_size": 3,
"activation": "relu",
"filter_scheme": "half",
"batch_norm": "True",
"dropout_prob": 0,
"dense_neurons": 128,
"lr": 1.4e-3,
"batch_size": 32,
"img_size": 224,
"num_classes": 10
```
**Accuracy:**
```
Test Accuracy: 36.00%
Train Accuracy: 47.5%
Validation Accuracy: 34.7%
```
```bash
python Q2.py
```
---
## Q4:
10×3 grid containing sample images from the test data & predictions made by best model parameter:
```
epoch : 10
test_acc : 33.33
train_acc : 45.1
val_acc : 38.1
```
```bash
python Q4.py
```
## Prediction on the Test dataset:
### Model trained on best parameter from scratch:
![Prediction Grid](/partA/prediction_grid.png)
---

- `epoch=9-step=1250.ckpt`: Automatically saved model checkpoint at step 1250 of epoch 9.
- `best-cnn.ckpt`: Best-performing model saved based on validation metrics from the best parameter.
- `prediction_grid.png` – A grid visualization of the model's predictions on sample test images from the best-cnn check point.
---