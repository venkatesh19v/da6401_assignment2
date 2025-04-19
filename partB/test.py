import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from FineTune import FineTune 
import random
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from PIL import Image
import wandb
from torchvision.transforms.functional import to_pil_image

wandb.init(project="DA6401_Assignment2", name="vit_b_16_test_logging")

MODEL_NAME = "vit_b_16"
STRATEGY = "freeze_until_layer4"
CKPT_PATH = "vit_b_16_freeze_until_layer4_sweep.ckpt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_ds = ImageFolder("/inaturalist_12K/val", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
class_names = test_ds.classes

model = FineTune.load_from_checkpoint(
    CKPT_PATH,
    num_classes=10,
    class_names=class_names,
    model_name=MODEL_NAME,
    strategy=STRATEGY,
    lr=1e-3 
)
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

# Choose 30 random indices from the test set
random_indices = random.sample(range(len(test_ds)), 30)
subset = Subset(test_ds, random_indices)
subset_loader = DataLoader(subset, batch_size=30, shuffle=False)

# Get batch of 30 images
images, labels = next(iter(subset_loader))
images = images.to(device)
labels = labels.to(device)

# Predict
with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# Move tensors to CPU for plotting
images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()

# Grid plot
fig, axes = plt.subplots(10, 3, figsize=(12, 25))
axes = axes.flatten()

unnorm = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

wandb_images = []

for i in range(30):
    img = unnorm(images[i])              
    img = torch.clamp(img, 0, 1)
    pil_img = to_pil_image(img)           

    axes[i].imshow(pil_img)
    axes[i].axis('off')
    axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", fontsize=8)

    caption = f"True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}"
    wandb_images.append(wandb.Image(pil_img, caption=caption))

# Save locally
plt.tight_layout()
output_path = "random_test_predictions_grid.png"
plt.savefig(output_path)
plt.close()
print(f"Saved test prediction grid to {output_path}")

wandb.log({"random_test_predictions": wandb_images})
wandb.finish()