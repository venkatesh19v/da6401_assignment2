import os
import torch
import wandb
from CNN import CNN
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedShuffleSplit

def prepare_data_loaders(batch_size, img_size):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    full = datasets.ImageFolder("inaturalist_12K/train", transform=tfm)
    labels = [label for _, label in full.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(full.samples, labels))
    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "num_filters":    {"values": [32, 64, 128]},
        "kernel_size":    {"values": [3, 5]},
        "activation":     {"values": ["relu", "gelu", "silu", "mish"]},
        "filter_scheme":  {"values": ["same", "double", "half"]},
        "batch_norm":     {"values": [True, False]},
        "dropout_prob":   {"values": [0.0, 0.2, 0.3]},
        "dense_neurons":  {"values": [64, 128, 256]},
        "lr":             {"min": 1e-4, "max": 1e-2},
        "batch_size":     {"values": [32]},
        "img_size":       {"value": 224},
        "num_classes":    {"value": 10}
    }
}

def train():
    with wandb.init() as run:
        config = dict(wandb.config)
        try:

            run_name = f"nf_{config['num_filters']}_ks_{config['kernel_size']}_{config['activation']}_{config['filter_scheme']}_dp_{config['dropout_prob']}_dn_{config['dense_neurons']}_lr_{config['lr']:.1e}_bs_{config['batch_size']}"
            run.name = run_name
            print(run_name)
            train_loader, val_loader = prepare_data_loaders(config["batch_size"], config["img_size"])

            model = CNN(config)
            wandb_logger = WandbLogger(project="DA6401_Assignment2")
            trainer = pl.Trainer(
                max_epochs=10,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                logger=wandb_logger,
                log_every_n_steps=10,
                enable_checkpointing=False
            )
            trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"[ERROR] Run failed: {e}")

sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment2")
wandb.agent(sweep_id, train)  # change count as needed