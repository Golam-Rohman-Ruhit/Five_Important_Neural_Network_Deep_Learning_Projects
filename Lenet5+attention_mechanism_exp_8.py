import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time
import os
import torch.nn.functional as F


# ==========================================
# 1. Attention Module (Spatial Attention)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Attention map generation
        out = self.conv1(x_cat)
        attention_map = self.sigmoid(out)

        return x * attention_map, attention_map


# ==========================================
# 2. Model Definitions
# ==========================================

# --- Baseline: LeNet-5 ---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Advanced: LeNet-5 + Attention ---
class AttnLeNet5(nn.Module):
    def __init__(self):
        super(AttnLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)

        # Attention Module inserted after first pooling
        self.attention = SpatialAttention(kernel_size=7)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        # Apply Attention
        x, attn_map = self.attention(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, attn_map


# ==========================================
# 3. Training Utilities
# ==========================================

def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(images)
        if isinstance(output, tuple):
            output = output[0]

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ==========================================
# 4. Main Execution
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (Ryzen CPU will take approx 15-20 mins for best results)")

    # --- PRO CONFIGURATION ---
    BATCH_SIZE = 64
    EPOCHS = 15  # Increased for convergence
    LR = 0.001

    # Data Augmentation (Helps reach >99%)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),  # Randomly rotate images +/- 10 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading Data...")
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # --- 1. Train Standard LeNet-5 ---
    print("\n[1/2] Training Standard LeNet-5 (Baseline)...")
    model_std = LeNet5().to(device)
    opt_std = optim.Adam(model_std.parameters(), lr=LR)
    scheduler_std = optim.lr_scheduler.StepLR(opt_std, step_size=5, gamma=0.1)  # Decays LR
    crit = nn.CrossEntropyLoss()

    acc_std = []
    for ep in range(EPOCHS):
        loss = train_model(model_std, device, train_loader, opt_std, crit)
        acc = evaluate_model(model_std, device, test_loader)
        scheduler_std.step()  # Update LR
        acc_std.append(acc)
        print(f"  Epoch {ep + 1}/{EPOCHS}: Loss={loss:.4f}, Acc={acc:.2f}%")

    # --- 2. Train Attention LeNet-5 ---
    print("\n[2/2] Training Attention LeNet-5 (Proposed Method)...")
    model_attn = AttnLeNet5().to(device)
    opt_attn = optim.Adam(model_attn.parameters(), lr=LR)
    scheduler_attn = optim.lr_scheduler.StepLR(opt_attn, step_size=5, gamma=0.1)  # Decays LR

    acc_attn = []
    for ep in range(EPOCHS):
        loss = train_model(model_attn, device, train_loader, opt_attn, crit)
        acc = evaluate_model(model_attn, device, test_loader)
        scheduler_attn.step()  # Update LR
        acc_attn.append(acc)
        print(f"  Epoch {ep + 1}/{EPOCHS}: Loss={loss:.4f}, Acc={acc:.2f}%")

    # Save Best Model
    torch.save(model_attn.state_dict(), 'attn_lenet_mnist_best.pth')
    print("\nBest Model Saved Successfully!")

    # ==========================================
    # 5. Generate Figures (No CV2 dependency)
    # ==========================================

    # Figure 1: Comparison Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), acc_std, 'r--o', label='Standard LeNet-5')
    plt.plot(range(1, EPOCHS + 1), acc_attn, 'g-o', label='Attention LeNet-5')
    plt.title('Accuracy Comparison: LeNet-5 vs. Attention LeNet-5')
    plt.xlabel('Epochs');
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True);
    plt.legend()
    plt.savefig('comparison_curve.png')
    plt.show()

    # Figure 2: Attention Map Visualization
    print("\nGenerating Attention Heatmaps...")
    model_attn.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)

    outputs, attn_maps = model_attn(images)
    _, preds = torch.max(outputs, 1)

    # Plot top 5 samples
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # Original Image
        img = images[i].cpu().squeeze().numpy()
        img = (img * 0.5) + 0.5

        # Process Attention Map (Upscale to 32x32)
        attn = attn_maps[i].unsqueeze(0)
        attn_resized = F.interpolate(attn, size=(32, 32), mode='bilinear', align_corners=False)
        attn_numpy = attn_resized.squeeze().cpu().detach().numpy()

        # Row 1: Original
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(f"True: {labels[i].item()}")
        axs[0, i].axis('off')

        # Row 2: Heatmap Overlay
        axs[1, i].imshow(img, cmap='gray')
        axs[1, i].imshow(attn_numpy, cmap='jet', alpha=0.5)
        axs[1, i].set_title(f"Attention (Pred: {preds[i].item()})")
        axs[1, i].axis('off')

    plt.suptitle("Visualizing Attention Weights: Where the Model Focuses")
    plt.savefig('attention_visualization.png')
    plt.show()

    # Figure 3: Confusion Matrix
    print("Generating Confusion Matrix...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model_attn(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Attention LeNet-5 Confusion Matrix (Acc: {acc_attn[-1]:.2f}%)")
    plt.savefig('confusion_matrix.png')
    plt.show()

    print(f"\nFinal Attention Model Accuracy: {acc_attn[-1]:.2f}%")
    print("Optimization Complete.")


if __name__ == '__main__':
    main()