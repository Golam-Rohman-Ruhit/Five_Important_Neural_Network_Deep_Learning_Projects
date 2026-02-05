import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def main():
    # ১. ডিভাইস কনফিগারেশন
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # হাইপার-প্যারামিটার (Hyper-parameters)
    num_classes = 10
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # ২. ডেটাসেট প্রসেসিং (LeNet-5 এর জন্য 32x32 সাইজ দরকার)
    # MNIST ইমেজ ২৮x২৮, তাই আমরা রিসাইজ করে ৩২x৩২ করব
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # ৩. LeNet-5 আর্কিটেকচার (Classic CNN)
    class LeNet5(nn.Module):
        def __init__(self, num_classes):
            super(LeNet5, self).__init__()
            # Layer 1: Convolutional Layer
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            # Layer 2: Convolutional Layer
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            # Layer 3: Fully Connected Layers
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Tanh(),
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes)
            )

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.fc(out)
            return out

    model = LeNet5(num_classes).to(device)
    print("LeNet-5 Model Created")

    # ৪. লস এবং অপটিমাইজার (Adam ব্যবহার করা হচ্ছে টিচারের ফিডব্যাক অনুযায়ী)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    accuracy_history = []

    # ৫. ট্রেনিং লুপ
    print("\nStarting Training (LeNet-5)...")
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / total_step
        loss_history.append(avg_loss)

        # একুরেসি চেক
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            accuracy_history.append(acc)
        model.train()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    print("Training Finished!")

    # ==========================================
    # ৬. রিপোর্ট গ্রাফস (Report Visualization)
    # ==========================================

    # গ্রাফ ১: Loss & Accuracy Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='purple', label='Training Loss')
    plt.title('LeNet-5 Training Loss')
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', color='green', label='Test Accuracy')
    plt.title('LeNet-5 Accuracy Curve')
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)');
    plt.grid(True);
    plt.legend()
    plt.tight_layout()
    plt.show()

    # গ্রাফ ২: Confusion Matrix
    print("Generating Confusion Matrix...")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Greens, ax=ax)
    plt.title("LeNet-5 Confusion Matrix")
    plt.show()

    # গ্রাফ ৩: Failure Analysis (টিচারের নতুন রিকোয়ারমেন্ট)
    print("Visualizing Misclassified Samples (Failure Analysis)...")
    misclassified_imgs = []
    misclassified_preds = []
    true_lbls = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # ভুলগুলো খুঁজে বের করা
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

    # ভুলের ছবিগুলো প্লট করা
    fig = plt.figure(figsize=(10, 4))
    for i in range(len(misclassified_imgs)):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        img = misclassified_imgs[i].squeeze().numpy()
        img = img * 0.5 + 0.5  # un-normalize
        ax.imshow(img, cmap='gray')
        # লাল টাইটেল (ভুল উত্তর)
        ax.set_title(f"P:{misclassified_preds[i]} (T:{true_lbls[i]})", color='red', fontweight='bold')

    plt.suptitle("Misclassified Samples (Failure Cases)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()