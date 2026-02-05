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


# ==========================================
# 1. ResNet Architecture (Optimized for MNIST)
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv1 -> BatchNorm -> ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Objective: Understand impact of BatchNorm
        self.relu = nn.ReLU(inplace=True)

        # Conv2 -> BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # The "Skip Connection" allows gradients to flow effectively
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNetMNIST(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetMNIST, self).__init__()
        self.in_channels = 64

        # Modified Initial Layer for small MNIST images (32x32)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet Layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNetMNIST(ResidualBlock, [2, 2, 2, 2])


# ==========================================
# 2. Main Execution Block (CPU Optimized)
# ==========================================

def main():
    # Force CPU Usage
    device = torch.device('cpu')
    print("=" * 60)
    print(f"Device Setting: Using CPU (Ryzen 5000 Series is powerful enough!)")
    print("Note: Training might take 10-15 mins. Please wait...")
    print("=" * 60)

    # Hyperparameters
    # 3 Epochs is sufficient for ResNet to reach >99% on MNIST
    num_epochs = 3
    batch_size = 64
    learning_rate = 0.001

    # Data Preprocessing (Resize to 32x32 for ResNet)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("\n[Step 1/4] Loading and Transforming Data...")
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = ResNet18().to(device)
    print("[Step 2/4] ResNet-18 Model Initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []

    print(f"\n[Step 3/4] Starting Training for {num_epochs} Epochs...")
    start_time = time.time()
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Progress Indicator (More frequent updates for CPU reassurance)
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Batch [{i + 1}/{total_step}] | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / total_step
        train_losses.append(avg_loss)

        # Validation Phase
        print(f"Validating Epoch {epoch + 1}...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        test_accuracies.append(acc)

        print(f"==> COMPLETED Epoch [{epoch + 1}/{num_epochs}] | Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    total_time = (time.time() - start_time) / 60
    print(f"\nTraining Finished in {total_time:.2f} minutes.")

    # ==========================================
    # SAVE MODEL
    # ==========================================
    save_path = 'resnet_mnist_cpu.pth'
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully to: {os.path.abspath(save_path)}")

    # ==========================================
    # [Step 4/4] Generate Figures for Report
    # ==========================================

    # Figure 1: Loss & Accuracy Curve (Critical for "Convergence" Objective)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Training Loss')
    plt.title('ResNet Training Convergence')
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, 'g-o', label='Test Accuracy')
    plt.title('ResNet Classification Accuracy')
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)');
    plt.grid(True);
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 2: Confusion Matrix (Critical for "Evaluate Performance" Objective)
    print("Generating Confusion Matrix...")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("ResNet-18 Confusion Matrix")
    plt.show()

    # Figure 3: Failure Analysis (To show robustness compared to LeNet)
    print("Finding Misclassified Samples...")
    misclassified_imgs = []
    misclassified_preds = []
    true_lbls = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            mask = predicted != labels
            if mask.sum() > 0:
                wrong_idx = mask.nonzero(as_tuple=True)[0]
                for idx in wrong_idx:
                    if len(misclassified_imgs) < 10:
                        misclassified_imgs.append(images[idx].cpu())
                        misclassified_preds.append(predicted[idx].cpu().item())
                        true_lbls.append(labels[idx].cpu().item())
            if len(misclassified_imgs) >= 10:
                break

    if len(misclassified_imgs) > 0:
        fig = plt.figure(figsize=(10, 4))
        for i in range(len(misclassified_imgs)):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            img = misclassified_imgs[i].squeeze().numpy()
            img = img * 0.5 + 0.5
            ax.imshow(img, cmap='gray')
            ax.set_title(f"P:{misclassified_preds[i]} (T:{true_lbls[i]})", color='red', fontweight='bold')
        plt.suptitle("ResNet Failure Analysis (Hard Samples)")
        plt.tight_layout()
        plt.show()
    else:
        print("Amazing! Perfect accuracy on the checked batch.")

    print(f"\nFinal ResNet Accuracy: {test_accuracies[-1]:.2f}%")
    print("(Note: This should be higher than LeNet's 98.4%)")


if __name__ == '__main__':
    main()