import torch

# 数据集路径
DATA_ROOT = './data'

# 训练超参数
BATCH_SIZE = 128          # 根据GPU内存调整
EPOCHS = 50
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# 模型保存路径
SAVE_PATH = './best_model.pth'

# 设备自动选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理参数（ImageNet的均值和标准差用于预训练模型）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 类别名称（CIFAR-10）
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']