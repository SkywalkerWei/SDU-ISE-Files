import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 1. 定义LeNet-5模型 ---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 卷积层
        # 输入: 1x32x32, 输出: 6x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # S2: 池化层 (下采样)
        # 输入: 6x28x28, 输出: 6x14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: 卷积层
        # 输入: 6x14x14, 输出: 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # S4: 池化层 (下采样)
        # 输入: 16x10x10, 输出: 16x5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: 卷积层，但在现代实现中通常被看作全连接层
        # 输入: 16x5x5, 输出: 120x1x1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        # 将多维张量展平
        self.flatten = nn.Flatten()
        
        # F6: 全连接层
        # 输入: 120, 输出: 84
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        
        # Output: 全连接层 (输出层)
        # 输入: 84, 输出: 10 (对应0-9十个数字)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # C1 -> ReLU -> S2
        x = self.pool1(self.relu(self.conv1(x)))
        # C3 -> ReLU -> S4
        x = self.pool2(self.relu(self.conv2(x)))
        # C5 -> ReLU
        x = self.relu(self.conv3(x))
        # 展平
        x = self.flatten(x)
        # F6 -> ReLU
        x = self.relu(self.fc1(x))
        # Output Layer
        x = self.fc2(x)
        return x

# --- 2. 设置超参数和设备 ---
EPOCHS = 20
BATCH_SIZE = 128 # Batch Size 是一次扔进网络里一起计算、并共同决定这次参数更新方向的样本数量。
LEARNING_RATE = 0.001

# Epoch（轮次/周期）：整个训练数据集完整地通过神经网络一次的过程。
# 例如，如果训练集有 60,000 张图片，1个 epoch 就是所有 60,000 张图片都参与了一次训练。
# Iteration（迭代）：完成一个批大小（Batch）的数据训练的过程。在这个过程中，需要进行 60,000 / 128 ≈ 469 次 Iteration。
# Batch Size（批大小）：一次迭代中使用的样本数量。在每一次 Iteration 中，都会随机抽取 128 张图片（一个Batch），将它们输入网络，计算这128张图片的平均损失（Loss），然后根据这个平均损失进行一次反向传播和参数更新。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 3. 数据预处理和加载 ---
# LeNet-5的输入是32x32，而MNIST是28x28，所以需要进行填充(Padding)
transform = transforms.Compose([
    transforms.Pad(2),      # 左右各填充2个像素，上下各填充2个像素，(28+2+2) = 32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 归一化
])

# 下载和加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. 初始化模型、损失函数和优化器 ---
model = LeNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 训练模型 ---
def train_model():
    start_time = time.time()
    # 用于记录训练过程中的损失和准确率，方便后续可视化
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train() # 设置为训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # 在每个epoch后进行验证
        model.eval() # 设置为评估模式
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Test Accuracy: {val_acc:.2f}%")
    return history

# --- 6. 可视化函数 ---
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training history plot to training_history.png")

def visualize_predictions():
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # 将图像和标签移回CPU以便于matplotlib显示
    images = images.cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    for i in range(25): # 显示25个样本的预测结果
        plt.subplot(5, 5, i+1)
        # 从 (1, 32, 32) 转换为 (32, 32)
        img = np.squeeze(images[i]) 
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        pred_label = predicted[i].item()
        true_label = labels[i].item()
        
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        
    plt.tight_layout()
    plt.savefig("prediction_examples.png")
    print("Saved prediction examples plot to prediction_examples.png")


# --- 7. 执行主流程 ---
if __name__ == '__main__':
    training_history = train_model()
    plot_history(training_history)
    visualize_predictions()