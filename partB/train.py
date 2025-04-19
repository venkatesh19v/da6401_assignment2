import os
import torch
import wandb
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from FineTuneModel import FineTune

# Specific model and strategy to train
MODEL_NAME = "resnet50"  # Specify the model you want to train
STRATEGY = "freeze_all"  # Specify the strategy you want to use
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10

# Dataset transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for all models for simplicity
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load and split dataset
full_train_ds = ImageFolder("/home/venkatesh/Documents/IDL/Assignment_2/As2/da6401_assignment2/inaturalist_12K/train", transform=transform)
train_len = int(0.8 * len(full_train_ds))
val_len = len(full_train_ds) - train_len
generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(full_train_ds, [train_len, val_len], generator)
test_ds = ImageFolder("/home/venkatesh/Documents/IDL/Assignment_2/As2/da6401_assignment2/inaturalist_12K/val", transform=transform)
class_names = test_ds.classes

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def has_trainable_params(model):
    return any(p.requires_grad for p in model.parameters())

# Start wandb run
run = wandb.init(
    project="DA6401_Assignment2",
    name=f"{MODEL_NAME}_{STRATEGY}",
    config={
        "model": MODEL_NAME,
        "strategy": STRATEGY,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }
)

# Initialize the model with the chosen strategy
model = FineTune(
    num_classes=10,
    class_names=class_names,
    model_name=MODEL_NAME,
    strategy=STRATEGY,
    lr=LR
)

# Initialize model checkpointing
checkpoint_cb = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    filename=f"{MODEL_NAME}_{STRATEGY}"
)

# Initialize the Wandb logger
wandb_logger = WandbLogger(project="DA6401_Assignment2", log_model=True)

# After initializing `model = FineTune(...)`
if has_trainable_params(model):
    trainer = Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices="auto",
    )
else:
    print(f"Warning: No trainable parameters in {MODEL_NAME} with strategy {STRATEGY}. Running on CPU.")
    trainer = Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        accelerator="cpu",  # Force CPU to avoid DDP crash
        devices=1
    )

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model and log metrics
test_metrics = trainer.test(model, test_loader, ckpt_path=checkpoint_cb.best_model_path)[0]

# Log test metrics manually
wandb.log({
    "final_test_acc": test_metrics["test_acc"],
    "final_test_loss": test_metrics["test_loss"]
})

# Finish the wandb run
wandb.finish()
