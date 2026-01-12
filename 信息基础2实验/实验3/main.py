# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import build_resnet18
from utils import plot_results

def main():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 定义数据预处理和增强
    # 对训练集进行数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 对测试集只进行标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)
    
    # 4. 初始化模型、损失函数和优化器
    net = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 使用学习率调度器，在训练过程中动态调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 5. 训练和评估循环
    num_epochs = 100 # 可根据需要调整
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # --- 训练 ---
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix(loss=train_loss/(batch_idx+1), acc=f'{100.*correct/total:.2f}%')
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(100. * correct / total)

        # --- 验证 ---
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        progress_bar_val = tqdm(testloader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar_val):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar_val.set_postfix(loss=test_loss/(batch_idx+1), acc=f'{100.*correct/total:.2f}%')
        
        acc = 100. * correct / total
        history['val_loss'].append(test_loss / len(testloader))
        history['val_acc'].append(acc)

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} Done: Train Acc: {history['train_acc'][-1]:.2f}%, Val Acc: {acc:.2f}%")


    print('Finished Training')

    # 6. 保存模型和可视化结果
    torch.save(net.state_dict(), 'results/resnet18_cifar10.pth')
    print("Model saved to 'results/resnet18_cifar10.pth'")
    
    plot_results(history)

if __name__ == '__main__':
    main()