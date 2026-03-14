import torch.nn as nn
import torchvision.models as models
import config

def get_resnet50(num_classes=10, pretrained=False):
    """
    创建ResNet50模型
    Args:
        num_classes: 分类数量（CIFAR-10为10）
        pretrained: 是否加载ImageNet预训练权重
    Returns:
        model: 修改后的ResNet50
    """
    # 加载预训练模型（如果pretrained=True）
    model = models.resnet50(pretrained=pretrained)
    
    # 获取原全连接层输入特征数
    in_features = model.fc.in_features
    
    # 替换全连接层为新的线性层（输出10类）
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# if __name__ == "__main__":
#     # 简单测试模型输出
#     model = get_resnet50(pretrained=False)
#     print(model)
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params/1e6:.2f}M")