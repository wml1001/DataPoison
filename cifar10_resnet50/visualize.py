import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import get_dataloaders
from model import get_resnet50
import config

# 核心修改1：给imshow函数添加ax参数，所有绘图操作绑定到指定子图
def imshow(img, ax, title=None):
    """显示图像（去标准化），绑定到指定的axes子图对象"""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array(config.NORM_MEAN)
    std = np.array(config.NORM_STD)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    # 用子图ax的imshow，而非全局plt.imshow，确保画到对应子图
    ax.imshow(img)
    if title is not None:
        ax.set_title(title)
    # 关闭当前子图的坐标轴，而非全局
    ax.axis('off')

def visualize_predictions(model, test_loader, device, num_images=8):
    """随机选取一批图像，显示真实标签和预测标签"""
    model.eval()
    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    # 防止batch_size小于要显示的图片数量，做个兜底
    images, labels = images[:num_images], labels[:num_images]
    real_show_num = len(images)
    
    # 预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # 转为CPU以便绘图
    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()
    
    # 绘制图像
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    # 核心修改2：循环中给每个子图传入对应的ax对象
    for i in range(real_show_num):
        ax = axes[i]
        true_label = config.CLASSES[labels[i]]
        pred_label = config.CLASSES[preds[i]]
        color = 'green' if preds[i] == labels[i] else 'red'
        # 传入ax，把图像画到对应子图
        imshow(images[i], ax=ax)
        # 设置子图标题
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12)
    
    # 处理多余的空坐标轴（如果实际图片数不足8个）
    for i in range(real_show_num, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', bbox_inches='tight', dpi=150)  # 优化保存效果
    plt.show()

def plot_training_curves(history):
    """
    绘制训练过程中的损失和准确率曲线
    history: 字典，包含'train_loss', 'val_loss', 'train_acc', 'val_acc'列表
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', bbox_inches='tight', dpi=150)
    plt.show()

def main():
    device = config.DEVICE
    
    # 加载测试集
    _, test_loader = get_dataloaders()
    
    # 加载模型
    model = get_resnet50(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
    print("Model loaded.")
    
    # 可视化预测
    visualize_predictions(model, test_loader, device, num_images=8)
    
    # 如果需要绘制训练曲线，可以传入history（需要在训练时保存）
    # 示例：history = np.load('history.npy', allow_pickle=True).item()
    # plot_training_curves(history)

if __name__ == "__main__":
    main()