# model.py
import torch
from torchvision import models

def get_resnet50(num_classes=10, pretrained=False):
    """返回 ResNet50 并替换最后一层为指定类别数"""
    try:
        WeightsEnum = models.ResNet50_Weights
        weights = WeightsEnum.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model
