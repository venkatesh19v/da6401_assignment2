import os
import wandb
import torch
import random
from CNN import CNN
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

project_name='DA6401_Assignment2'
# wandb.login()
# wandb.init(project=project_name)

def train_cnn(config):
    wandb.init(project="DA6401_Assignment2", config=config)
    config = dict(wandb.config)

    # Prepare data loaders
    def prepare_data_loaders(batch_size, img_size):
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        full = datasets.ImageFolder("/home/venkatesh/Documents/IDL/Assignment_2/As2/da6401_assignment2/inaturalist_12K/train", transform=tfm)
        labels = [y for _, y in full.samples]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(full.samples, labels))
        train_ds = Subset(full, train_idx)
        val_ds = Subset(full, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, val_loader

    train_loader, val_loader = prepare_data_loaders(config["batch_size"], config["img_size"])

    # Model + logger + trainer
    model = CNN(config)
    wandb_logger = WandbLogger(project="DA6401_Assignment2", log_model="all")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()


#Hyperparameters
config = {
    "num_filters": 32,
    "kernel_size": 3,
    "activation": "relu",
    "filter_scheme": "same",
    "batch_norm": True,
    "dropout_prob": 0.2,
    "dense_neurons": 128,
    "lr": 1e-3,
    "batch_size": 64,
    "img_size": 224,
    "num_classes": 10
}

train_cnn(config)

test_transform = transforms.Compose([
    transforms.Resize((config["img_size"], config["img_size"])),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder("inaturalist_12K/val", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
class_names = test_dataset.classes

ckpt_path = "epoch=9-step=1250.ckpt" 
model = CNN.load_from_checkpoint(ckpt_path, config=config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
test_acc = (all_preds == all_labels).float().mean().item()

print(f"\nTest Accuracy: {test_acc:.4f}")

# wandb.init(project=project_name)
sample_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
images, labels = next(iter(sample_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(1)

fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(random.sample(range(len(images)), 5)):
    img = images[idx].cpu().permute(1, 2, 0)
    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(f"Pred: {class_names[preds[idx]]}\nTrue: {class_names[labels[idx]]}")
plt.savefig("sample_predictions.png")
wandb.log({"Sample Predictions": wandb.Image(fig)})