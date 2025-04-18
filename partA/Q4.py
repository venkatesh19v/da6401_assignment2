import wandb
import torch
from CNN import CNN 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchvision import datasets, transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

wandb.init(project="DA6401_Assignment2", name="final-grid-visualization", reinit=True)

pl.seed_everything(42)

# Best config from sweep
config = {
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
    "num_classes": 10,
}

transform = transforms.Compose([
    transforms.Resize((config["img_size"], config["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_train = datasets.ImageFolder(root="inaturalist_12K/train", transform=transform)
test_dataset = datasets.ImageFolder(root="inaturalist_12K/val", transform=transform)

# Stratified 80/20 split
targets = [s[1] for s in full_train.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
train_subset = Subset(full_train, train_idx)
val_subset = Subset(full_train, val_idx)

train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Saving best model
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    filename="best-cnn",
    save_top_k=1,
    verbose=True
)

# Train
model = CNN(config)
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback],
    accelerator="auto"
)

trainer.fit(model, train_loader, val_loader)

# Load best model
best_model = CNN(config)
best_model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])

# Evaluate on test data
trainer.test(best_model, test_loader)

# 10×3 grid: Show predictions
def imshow(img_tensor):
    img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_tensor = img_tensor.clamp(0, 1)
    return np.transpose(img_tensor.numpy(), (1, 2, 0))

classes = test_dataset.classes
best_model.eval()

fig, axes = plt.subplots(10, 3, figsize=(10, 30))
with torch.no_grad():
    i = 0
    for images, labels in test_loader:
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)
        for j in range(images.size(0)):
            if i >= 30: break
            ax = axes[i // 3, i % 3]
            ax.imshow(imshow(images[j].cpu()))
            ax.set_title(f"Pred: {classes[preds[j]]}\nTrue: {classes[labels[j]]}", fontsize=9)
            ax.axis("off")
            i += 1
        if i >= 30:
            break

plt.tight_layout()
plt.savefig("prediction_grid.png")
wandb.log({"Prediction Grid": wandb.Image(fig)})

wandb.finish()
# plt.show()
