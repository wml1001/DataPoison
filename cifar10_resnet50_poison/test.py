import torch
import torch.nn as nn
from data_utils import get_dataloaders
from model import get_resnet50
import config

def test(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 加载测试集
    _, test_loader = get_dataloaders()  # 注意第二个返回值是test_loader
    
    # 构建模型并加载权重
    model = get_resnet50(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
    print("Model loaded successfully.")
    
    # 测试
    acc = test(model, test_loader, device)
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()