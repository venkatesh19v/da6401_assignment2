import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
def get_activation(name):
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish()
    }[name]

# 5-layer flexible CNN
class CNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(dict(config))
        m = config["num_filters"]
        k = config["kernel_size"]
        act = get_activation(config["activation"])
        use_bn = config["batch_norm"]
        do_p = config["dropout_prob"]

        # Create conv blocks
        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, k, padding=k // 2)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act)
            layers.append(nn.MaxPool2d(2))
            if do_p > 0:
                layers.append(nn.Dropout2d(do_p))
            return nn.Sequential(*layers)

        # Filter scheme: same, double, half
        if config["filter_scheme"] == "same":
            fs = [m] * 5
        elif config["filter_scheme"] == "double":
            fs = [m * (2 ** i) for i in range(5)]
        elif config["filter_scheme"] == "half":
            fs = [max(1, m // (2 ** i)) for i in range(5)]  # avoid 0
        else:
            raise ValueError("Invalid filter scheme")

        # 5 convolution blocks
        self.blocks = nn.Sequential(
            conv_block(3, fs[0]),
            conv_block(fs[0], fs[1]),
            conv_block(fs[1], fs[2]),
            conv_block(fs[2], fs[3]),
            conv_block(fs[3], fs[4]),
        )

        # Compute flatten size after convs
        dummy = torch.zeros(1, 3, config["img_size"], config["img_size"])
        flat = self.blocks(dummy).view(1, -1).size(1)

        # Dense layers
        self.fc1 = nn.Linear(flat, config["dense_neurons"])
        self.act_fc = act
        self.drop_fc = nn.Dropout(do_p)
        self.out = nn.Linear(config["dense_neurons"], config["num_classes"])
        self.lr = config["lr"]

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.act_fc(self.fc1(x))
        x = self.drop_fc(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
