# ===================================================================
# 实验一：常规神经网络函数逼近实验 (PyTorch GPU 版本)
# 本代码将按顺序完成两个任务：XOR拟合 与 数学函数拟合
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 全局设置 ---
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备已设置为 -> {device}")
print("="*60)

# --- 2. 任务一：XOR 逻辑门拟合 ---

def run_xor_experiment():
    print("任务一：开始执行 XOR 逻辑门拟合...")
    
    # --- 数据准备 ---
    # 定义XOR的输入和期望输出
    X_numpy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y_numpy = np.array([[0], [1], [1], [0]], dtype=np.float32)
    # 将NumPy数组转换为张量，并发送到GPU
    X = torch.from_numpy(X_numpy).to(device)
    Y = torch.from_numpy(Y_numpy).to(device)

    # --- 模型定义 ---
    # 定义一个适用于XOR分类任务的三层神经网络
    class XORNet(nn.Module):
        def __init__(self):
            super(XORNet, self).__init__()
            self.layer1 = nn.Linear(2, 4)  # 输入层到隐藏层 (2个输入节点, 4个隐藏节点)
            self.layer2 = nn.Linear(4, 1)  # 隐藏层到输出层 (4个隐藏节点, 1个输出节点)
        def forward(self, x):
            # 定义数据在网络中的流向（前向传播）
            x = torch.sigmoid(self.layer1(x)) # 隐藏层使用sigmoid激活函数
            x = torch.sigmoid(self.layer2(x)) # 输出层也使用sigmoid，将输出压缩到0-1之间
            return x
    model = XORNet().to(device) # 实例化模型并将其移动到GPU
    
    # --- 训练过程 ---
    # 定义损失函数（均方误差）和优化器（Adam，一种高效的梯度下降算法）
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    # 使用较高的学习率以解决XOR不收敛问题，学习率后续继续微调
    epochs = 10000
    loss_history = []

    for epoch in range(epochs):
        predictions = model(X) # 1. 前向传播：计算预测值
        loss = loss_function(predictions, Y) # 2. 计算损失
        # 3. 反向传播与优化
        optimizer.zero_grad() # 清空上一轮的梯度
        loss.backward()       # 计算当前梯度
        optimizer.step()      # 根据梯度更新网络权重

        loss_history.append(loss.item())
        if (epoch + 1) % 1000 == 0:
            # 将模型的输出（0-1之间的浮点数）转换为0或1的预测标签
            predicted_labels = (predictions > 0.5).float()
            # 计算预测正确的样本数量
            correct_predictions = (predicted_labels == Y).sum().item()
            # 计算准确率
            accuracy = (correct_predictions / Y.size(0)) * 100
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%")
        
    print("任务一训练完成。")

    # --- 结果展示与可视化 ---
    model.eval() # 将模型切换到评估模式
    with torch.no_grad(): # 在此代码块中不计算梯度，节省计算资源
        final_predictions = model(X)
    # 将结果从GPU移回CPU，并转换为NumPy数组以便打印
    final_predictions_numpy = final_predictions.cpu().numpy()

    print("\n--- XOR 任务最终预测结果 ---")
    for i in range(len(X_numpy)):
        print(f"Input: {X_numpy[i]} -> Exp: {Y_numpy[i][0]}, Prediction: {final_predictions_numpy[i][0]:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.title("Task 1: XOR Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE) Loss")
    plt.grid(True)
    plt.savefig("xor_loss_curve.png")
    print("Loss curve for Task 1 saved to xor_loss_curve.png")
    print("="*60)

# --- 3. 任务二：y = 1/sin(x) + 1/cos(x)函数拟合 ---

def run_math_function_experiment():
    print("任务二：开始执行函数 `y = 1/sin(x) + 1/cos(x)` 拟合...")
    
    # --- 数据准备 ---
    x_numpy = np.linspace(0.1, 1.4, 200, dtype=np.float32).reshape(-1, 1)
    y_numpy = (1/np.sin(x_numpy) + 1/np.cos(x_numpy)).astype(np.float32)
    X = torch.from_numpy(x_numpy).to(device)
    Y = torch.from_numpy(y_numpy).to(device)

    # --- 模型定义 ---
    # 定义一个适用于回归任务的三层神经网络
    class MathNet(nn.Module):
        def __init__(self):
            super(MathNet, self).__init__()
            self.layer1 = nn.Linear(1, 32)  # 输入层到隐藏层 (1个输入, 32个隐藏节点)
            self.layer2 = nn.Linear(32, 1)  # 隐藏层到输出层 (32个隐藏节点, 1个输出)
        def forward(self, x):
            x = torch.relu(self.layer1(x)) # 隐藏层使用ReLU激活函数，通常在回归任务中表现很好
            x = self.layer2(x) # 输出层为线性层，不使用激活函数，以输出任意实数值
            return x
    model = MathNet().to(device)

    # --- 训练过程 ---
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 40000
    loss_history = []

    for epoch in range(epochs):
        predictions = model(X) # 1. 前向传播：计算预测值
        loss = loss_function(predictions, Y) # 2. 计算损失
        optimizer.zero_grad() # 3. 反向传播，清空上一轮的梯度
        loss.backward() # 计算当前梯度
        optimizer.step() # 根据梯度更新网络权重
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("任务二训练完成。")
    
    # --- 结果展示与可视化 ---
    model.eval()
    with torch.no_grad():
        predicted_y_numpy = model(X).cpu().numpy() # 类似地，发到CPU上转换为numpy用于可视化

    print("\n--- 数学函数任务结果可视化 ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图：损失曲线
    ax1.plot(loss_history)
    ax1.set_title("Task 2: Math Function Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean Squared Error (MSE) Loss")
    ax1.grid(True)

    # Function Fit
    ax2.scatter(x_numpy, y_numpy, label="True Data", s=15, alpha=0.7)
    ax2.plot(x_numpy, predicted_y_numpy, color='red', linewidth=2.5, label="Neural Network Prediction")
    ax2.set_title("Task 2: Function Fit Comparison")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("math_function_results.png")
    print("Result plot for Task 2 saved to math_function_results.png")
    print("="*60)

# --- 4. 按顺序执行两个实验 ---
if __name__ == '__main__':
    run_xor_experiment()
    run_math_function_experiment()
    print("所有任务已执行完毕。")