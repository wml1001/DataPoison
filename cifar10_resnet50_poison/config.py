# config.py
import torch

DATA_ROOT = './data'
BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

SAVE_PATH = './best_model.pth'
POISON_SAVE_PATH = './poisoned_samples.pth'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
TARGET_CLASS = 5   # dog
SOURCE_CLASS = 2   # bird

# ===== Meta-poison hyperparams (safe defaults for memory) =====
NUM_POISON = 1000       # <=50 推荐。若机器内存大可以提高（如 200）
OUTER_ITER = 20        # 外循环次数（调试时设小）
INNER_STEPS = 5        # 内展开步数 K
POISON_LR = 0.1        # 对投毒样本更新步长
EPSILON = 16.0 / 255.0 # L_inf 约束
INNER_LR = 0.001        # 内部 theta_hat 更新步长

# Per-step batch size used inside meta (独立于训练 BATCH_SIZE)
META_BATCH_SIZE = 8    # 内循环使用的小批量，减小可节省显存

# EoT (Expectation over Transformations)
USE_EOT = False
EOT_T = 3

# Meta execution device options
USE_CPU_META = False   # 若 True，会在 CPU 上运行 meta_poison（慢，但防 OOM）
# 如果想把 meta 放到指定设备，例如 'cpu' 或 'cuda:0'，可以设置 META_DEVICE below:
META_DEVICE = torch.device("cpu") if USE_CPU_META else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 新增超参数
POISON_RATIO = 0.25              # 每批投毒比例（方案A）
CLEAN_SUBSET_SIZE = None         # 若使用方案B，可设定干净子集大小（None 表示自动计算）
META_BATCH_SIZE = 8             # 元训练内循环小批量大小

# Debug quick (for fast local testing)
DEBUG_QUICK = False
if DEBUG_QUICK:
    NUM_POISON = 8
    OUTER_ITER = 2
    EOT_T = 1
    META_BATCH_SIZE = 4