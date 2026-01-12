# utils.py

import matplotlib.pyplot as plt

def plot_results(history):
    """
    Generate plots for training and validation loss and accuracy.
    """
    # 创建一个2x1的子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # 调整布局并保存图像
    fig.tight_layout()
    plt.savefig('results/training_curves.png')
    print("Training curves plot saved to 'results/training_curves.png'")