import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# 动态选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])
# 下载训练和测试数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
# 加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 可视化部分数据
def plot_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.savefig("fashion_mnist_data.png")  # 保存到文件
    plt.close()  # 关闭图形
    # plt.show()


# 3. 构建卷积神经网络（CNN）模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型训练过程
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    loss_history = []
    acc_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return loss_history, acc_history


# 6. 可视化训练过程
def plot_training_history(loss_history, acc_history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("fashion_mnist_train.png")  # 保存到文件
    plt.close()  # 关闭图形
    # plt.show()


# 7. 评估模型
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# 9. 使用训练好的模型进行预测
def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)  # 移动到设备
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# plt.show()

if __name__ == '__main__':
    # 可视化数据集部分数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    plot_images(images, labels)

    # 实例化模型并移到设备
    model = CNN().to(device)
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 5. 训练模型
    num_epochs = 30

    # 调用训练
    loss_history, acc_history = train(model, train_loader, criterion, optimizer, num_epochs)
    # 可视化训练部分
    plot_training_history(loss_history, acc_history)

    # 使用测试集 评估模型
    evaluate(model, test_loader)

    # 保存模型
    torch.save(model.state_dict(), 'fashion_mnist_cnn_model.pth')

    # 测试预测功能
    sample_image = test_dataset[0][0].unsqueeze(0).to(device)  # 获取一张图像并增加一个维度（批次大小）
    predicted_label = predict(model, sample_image)

    # 可视化预测结果
    plt.imshow(test_dataset[0][0].squeeze(), cmap='gray')
    plt.title(f'Predicted: {predicted_label}, True: {test_dataset[0][1]}')
    plt.axis('off')
    plt.savefig("fashion_mnist_pre.png")  # 保存到文件
    plt.close()  # 关闭图形
