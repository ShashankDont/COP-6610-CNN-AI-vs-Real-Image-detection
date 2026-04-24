import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

'''
 Claude was used to implement the Grad-CAM portion of the code.
 I am not familar with Grad-CAM and was not able to learn it quick enough for this project
 so I used Claude to code that part. 
'''
# This checks if an NVIDIA GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Data Loading & Preprocessing
ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")

# Split the data into 80% training and 20% test
split = ds["train"].train_test_split(test_size=0.2)
train_ds, val_ds = split["train"], split["test"]

# Resize images to same size and normalize for better convergence
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Converts images to RGB
def preprocess(batch):
    batch["pixel_values"] = [transform(img.convert("RGB")) for img in batch["image"]]
    return batch

train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

# Collate function: tells the DataLoader how to stack individual samples into a batch tensor
def collate_fn(examples):
    return torch.stack([x["pixel_values"] for x in examples]), torch.tensor([x["label"] for x in examples])

# DataLoader: handles shuffling, batching, and parallel data loading
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn, pin_memory=True)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers: extract features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Fully connected layers: classify based on extracted features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)

# Optomizer and Loss function
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("Starting training...")
model.train()

for epoch in range(5):
    for i, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

# Evaluation 
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, lbls in val_loader:
        outputs = model(imgs.to(device))
        preds   = torch.argmax(outputs, dim=1)
        y_true.extend(lbls.tolist())
        y_pred.extend(preds.cpu().tolist())

# Print evaluation meterics
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=["AI", "Real"]))

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=["AI", "Real"], yticklabels=["AI", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Grad-CAM (Claude Code)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients   = None
        self.activations = None

        # Forward hook: capture feature maps as the forward pass runs
        target_layer.register_forward_hook(self._save_activation)
        # Backward hook: capture gradients during backprop
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(True)

        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop only for the target class score
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool gradients → per-channel importance weights α
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  

        # Weighted sum of feature maps + ReLU 
        cam = (weights * self.activations).sum(dim=1, keepdim=True) 
        cam = torch.relu(cam)

        # Normalise 
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def visualize_gradcam(model, img_tensor, label, pred, cam_heatmap, ax_orig=None, ax_cam=None):
    # Undo ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)

    # Smooth + upsample heatmap
    heatmap = cv2.GaussianBlur(cam_heatmap, (5, 5), sigmaX=0)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_colored = plt.cm.turbo(heatmap)[..., :3]
    overlay = (0.65 * img + 0.35 * heatmap_colored).clip(0, 1)

    correct = "✓" if label == pred else "✗"
    title   = f"{correct} True: {label} | Pred: {pred}"
    color   = "green" if label == pred else "red"

    # Original image
    ax_orig.imshow(img)
    ax_orig.set_title("Original", fontsize=8)
    ax_orig.axis("off")

    # Grad-CAM overlay
    ax_cam.imshow(overlay)
    ax_cam.set_title(title, fontsize=8, color=color)
    ax_cam.axis("off")


# Visualzing the Grad-CAM
target_layer = model.conv_layers[-3]   #
grad_cam = GradCAM(model, target_layer)
class_names = ["AI", "Real"]
n_samples = 8
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

model.eval()
sample_count = 0

with torch.no_grad():
    for imgs, lbls in val_loader:
        for i in range(imgs.size(0)):
            if sample_count >= n_samples:
                break

            img_t = imgs[i:i+1].to(device)
            lbl   = class_names[lbls[i].item()]

            with torch.enable_grad():
                heatmap = grad_cam.generate(img_t)

            pred = class_names[model(img_t).argmax(dim=1).item()]

            row      = sample_count // 2        
            col_orig = (sample_count % 2) * 2       
            col_cam  = col_orig + 1                  

            visualize_gradcam(model, img_t, lbl, pred, heatmap,
                              ax_orig=axes[row, col_orig],
                              ax_cam=axes[row, col_cam])
            sample_count += 1

        if sample_count >= n_samples:
            break

plt.suptitle("Original vs Grad-CAM", fontsize=14)
plt.tight_layout()
plt.show()