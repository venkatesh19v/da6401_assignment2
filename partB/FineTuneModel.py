# finetune_module.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchvision import models

class FineTune(pl.LightningModule):
    def __init__(self, num_classes, class_names, model_name="resnet18", strategy="freeze_all", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names
        self.example_images = []

        # Load model dynamically
        model_fn = getattr(models, model_name)
        self.model = model_fn(pretrained=True)

        # Resize input if required (e.g. Inception needs input size 299)
        self.input_size = 299 if model_name == "inception_v3" else 224

        # Strategy control
        if strategy == "freeze_all":
            for param in self.model.parameters():
                param.requires_grad = False
        elif strategy == "freeze_until_layer4":
            for name, param in self.model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif strategy == "unfreeze_all":
            for param in self.model.parameters():
                param.requires_grad = True

        # Replace classification head
        if "resnet" in model_name or "resnext" in model_name or "shufflenet" in model_name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif "vgg" in model_name or "alexnet" in model_name:
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif "inception" in model_name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif "efficientnet" in model_name:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        elif "vit" in model_name:
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
