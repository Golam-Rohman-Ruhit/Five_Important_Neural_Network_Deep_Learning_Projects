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

    # হাইপার-প্যারামিটার
    input_size = 784
    hidden_size = 512
    num_classes = 10
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # ২. ডেটাসেট প্রসেসিং
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # ৩. মডেল আর্কিটেকচার
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MLP, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            out = self.fc2(x)
            return out

    model = MLP(input_size, hidden_size, num_classes).to(device)

    # ৪. লস এবং অপটিমাইজার
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # গ্রাফ আঁকার জন্য ডেটা স্টোর করা
    loss_history = []
    accuracy_history = []

    # ৫. ট্রেনিং লুপ
    print("\nStarting Training...")
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # প্রতি ইপোকের গড় লস বের করা
        avg_loss = epoch_loss / total_step
        loss_history.append(avg_loss)

        # প্রতি ইপোক শেষে একুরেসি চেক করা (গ্রাফের জন্য)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            accuracy_history.append(acc)
        model.train()  # আবার ট্রেনিং মোডে ফেরা

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')

    print("Training Finished!")

    # ==========================================
    # ৬. রিপোর্ট জেনারেশন গ্রাফস (Experimental Results)
    # ==========================================

    # গ্রাফ ১: Loss Curve এবং Accuracy Curve
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='red', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', color='blue', label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()  # এই গ্রাফের স্ক্রিনশট নিবে রিপোর্টের জন্য

    # গ্রাফ ২: Confusion Matrix (Discussion এর জন্য)
    print("Generating Confusion Matrix...")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.show()  # এই গ্রাফের স্ক্রিনশট নিবে

    # গ্রাফ ৩: ভিজু্যলাইজেশন (Predictions)
    print("Visualizing Predictions...")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images_flatten = images.reshape(-1, 28 * 28).to(device)
    outputs = model(images_flatten)
    _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(10, 4))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        img = images[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        pred_label = predicted[i].item()
        true_label = labels[i].item()
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"P:{pred_label} (T:{true_label})", color=color)

    plt.tight_layout()
    plt.show()  # এই গ্রাফের স্ক্রিনশট নিবে


if __name__ == '__main__':
    main()