import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ১. LeNet-5 আর্কিটেকচার (আগের মতোই)
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
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


def train_model(optimizer_name, device, train_loader, test_loader):
    print(f"\nTraining with optimizer: {optimizer_name}...")

    # প্রতিবার নতুন মডেল ইনিশিয়ালাইজ করা (যাতে আগের ট্রেনিং এফেক্ট না থাকে)
    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # অপটিমাইজার সিলেক্ট করা
    learning_rate = 0.001
    if optimizer_name == 'SGD':
        # SGD সাধারণত একটু স্লো হয়, তাই মোমেন্টাম যোগ করলাম ফেয়ার কম্পারিজন এর জন্য
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    accuracy_history = []

    # ৫ ইপোকের ট্রেনিং লুপ
    num_epochs = 5
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

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
        accuracy_history.append(acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] -> Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    return loss_history, accuracy_history


def main():
    # ডিভাইস সেটআপ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ডেটা লোডিং (32x32 Resize সহ)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ৩টি অপটিমাইজার দিয়ে লুপ চালানো
    optimizers_list = ['SGD', 'RMSprop', 'Adam']
    results = {}

    for opt_name in optimizers_list:
        losses, accuracies = train_model(opt_name, device, train_loader, test_loader)
        results[opt_name] = {'loss': losses, 'acc': accuracies}

    print("\nAll training finished! Generating Comparison Plots...")

    # ==========================================
    # রিপোর্ট গ্রাফ: ৩টি অপটিমাইজারের তুলনা
    # ==========================================
    epochs = range(1, 6)

    plt.figure(figsize=(14, 6))

    # গ্রাফ ১: Training Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['SGD']['loss'], 'r-o', label='SGD')
    plt.plot(epochs, results['RMSprop']['loss'], 'b-s', label='RMSprop')
    plt.plot(epochs, results['Adam']['loss'], 'g-^', label='Adam')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # গ্রাফ ২: Test Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['SGD']['acc'], 'r-o', label='SGD')
    plt.plot(epochs, results['RMSprop']['acc'], 'b-s', label='RMSprop')
    plt.plot(epochs, results['Adam']['acc'], 'g-^', label='Adam')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ফাইনাল সামারি টেবিল প্রিন্ট করা (রিপোর্টের জন্য কাজে লাগবে)
    print("\nFinal Performance Summary:")
    print(f"{'Optimizer':<10} | {'Final Loss':<12} | {'Final Accuracy':<15}")
    print("-" * 45)
    for opt in optimizers_list:
        final_loss = results[opt]['loss'][-1]
        final_acc = results[opt]['acc'][-1]
        print(f"{opt:<10} | {final_loss:.4f}       | {final_acc:.2f}%")


if __name__ == '__main__':
    main()