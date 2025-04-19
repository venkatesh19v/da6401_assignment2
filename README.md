# DA6401 Assignment 2: CNN-Based Image Classification on iNaturalist Dataset

This project focuses on developing and fine-tuning Convolutional Neural Network (CNN) models for image classification tasks using the iNaturalist 12K dataset. It encompasses both training models from scratch and fine-tuning pre-trained models to classify images into 10 distinct animal categories.

**W&B Report**: [View Report](https://api.wandb.ai/links/venkatesh19v-indian-institute-of-technology-madras/c7kqplz1)

**Github PartA**: [View Git](https://github.com/venkatesh19v/da6401_assignment2/tree/main/partA)
**Github PartB**: [View Git](https://github.com/venkatesh19v/da6401_assignment2/tree/main/partB)
---

## Project Structure

```
.
├── partA
│   ├── best-cnn.ckpt
│   ├── CNN.py
│   ├── epoch=9-step=1250.ckpt
│   ├── inaturalist_12K
│   ├── Q1.py
│   ├── Q2.py
│   ├── Q4.py
│   └── Question1.ipynb
├── partB
│   ├── FineTuneModel.py
│   └── sweep.py
├── requirements.txt
└── README.md
```

- **partA/**: Contains code for training CNN models from scratch.  
- **partB/**: Includes scripts for fine-tuning pre-trained models using various strategies.  
- **README.md**: Provides an overview and setup instructions.


**Best Sweep Config for Scratch**
```json
"num_filters": 128,
"kernel_size": 3,
"activation": "relu",
"filter_scheme": "half",
"batch_norm": True,
"dropout_prob": 0,
"dense_neurons": 128,
"lr": 1.4e-3,
"batch_size": 32,
"img_size": 224,
"num_classes": 10
```

```
Test Accuracy: 33.33%
Train Accuracy: 45.1%
Validation Accuracy: 38.1%
```
---

## Models Explored

The project investigates several pre-trained models from the `torchvision` library:

- ViT B/16  
- EfficientNet V2-S  
- VGG16  
- ResNet50  
- ResNet18  
- AlexNet

---

## Fine-Tuning Strategies

Three distinct strategies were employed to fine-tune the pre-trained models:

- **Freeze All**: All layers are frozen except the final classification layer.  
- **Freeze Until Layer4**: Layers up to `'layer4'` are frozen; subsequent layers are fine-tuned.  
- **Unfreeze All**: All layers are unfrozen and fine-tuned.

---

## Results Summary

The performance of each model-strategy combination was evaluated on the test set. Below is a summary of the test accuracies:

| Model               | Freeze Until Layer4 | Freeze All | Unfreeze All |
|--------------------|---------------------|------------|--------------|
| **ViT B/16**        | **84.50%**          | 83.50%     | 24.60%       |
| EfficientNet V2-S   | 84.20%              | 72.80%     | 70.70%       |
| VGG16               | 77.00%              | 75.40%     | 28.40%       |
| ResNet50            | 72.50%              | 75.40%     | 51.10%       |
| ResNet18            | 70.50%              | 67.70%     | 54.50%       |
| AlexNet             | 62.10%              | 63.00%     | 10.00%       |

### Best Performing Configuration

```
Model     : ViT B/16
Strategy  : Freeze Until Layer4
Test Acc  : 84.50%
Val Acc   : 84.10%
```
---

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/venkatesh19v/da6401_assignment2.git
   cd da6401_assignment2
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the iNaturalist 12K Dataset**:

   Ensure the dataset is downloaded and placed in the following structure:

   ```
   partA/inaturalist_12K/
   ├── train/
   └── val/
   ```

   Each subdirectory should contain folders for each of the 10 classes.

5. **Run Training Scripts**:

   - **Training from Scratch**:

     ```bash
     python partA/Q4.py
     ```

   - **Fine-Tuning Pre-Trained Models**:

     ```bash
     python partB/sweep.py
     ```

---


## Insights

- Fine-tuning pre-trained models, especially using the **Freeze Until Layer4** strategy, significantly outperforms training from scratch.
- The **Vision Transformer (ViT B/16)** achieved the highest accuracy, showcasing the strength of transformer-based architectures in image classification tasks.
- **Freezing all layers except the last one** provides a good balance between computational efficiency and performance.

## Prediction on the Test dataset:
### Model trained on best parameter from scratch:
![Prediction Grid](/partA/prediction_grid.png)

### Model trained from Finetuned model:
![Prediction Grid](/partB/random_test_predictions_grid.png)

---

## Conclusion
Results after the Fine-tuning is drastically improved it is evident from the prediction uploaded. ref. Fig. 1 and Fig. 2 Fine-tuning a pre-trained ViT model using a partial freezing strategy leads to:
 - Higher accuracy
 - Faster convergence
 - Better generalization compared to training a small CNN from scratch.
With a thoughtful strategy like freeze_until_layer4, it’s possible to balance performance and computational cost effectively.

---