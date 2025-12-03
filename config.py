import torch
import numpy as np
import medmnist

# ==========================================
# 1. 实验控制中心
# ==========================================
DATASET_NAME = 'FASHION_MNIST'
MODEL_TYPE = 'DFM_FNCN'
EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'

# ==========================================
# 2. 数据集配置
# ==========================================
DATASET_CONFIGS = {
    'FASHION_MNIST': {
        'n_classes': 10, 'in_channels': 1,
        'class_names': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
        'target_size': (28, 28)
    },
    'SVHN': {
        'n_classes': 10, 'in_channels': 3,
        'class_names': [str(i) for i in range(10)],
        'target_size': (28, 28)
    },
    'BLOOD_MNIST': {
        'n_classes': 8, 'in_channels': 3,
        'class_names': list(medmnist.INFO['bloodmnist']['label'].values()),
        'target_size': (28, 28)
    }
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"未知的 DATASET_NAME: {DATASET_NAME}")

CURRENT_DATA_CONFIG = DATASET_CONFIGS[DATASET_NAME]
N_CLASSES = CURRENT_DATA_CONFIG['n_classes']
IN_CHANNELS = CURRENT_DATA_CONFIG['in_channels']
CLASS_NAMES = CURRENT_DATA_CONFIG['class_names']
TARGET_SIZE = CURRENT_DATA_CONFIG['target_size']

# ==========================================
# 3. 全局训练配置
# ==========================================
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
SEED = 42
DATA_ROOT = './data'

# ==========================================
# 4. DFM-FNCN 模糊模型特定配置
# ==========================================
MAX_RULES = 100

# [重要修正] 恢复极小的阈值，因为我们现在使用的是乘法逻辑
# exp(-50) 约等于 1.9e-22，适合 128 维特征的连乘
PHI_TH = np.exp(-40)
INIT_SIGMA = 1.1

# [创新点 1] 模糊层改进: Attention Aggregation
# True: 使用加权乘积 (Weighted Product)
# False: 使用原始连乘 (Standard Product)
USE_ATTENTION = True

# ==========================================
# 5. 传统 CNN 配置
# ==========================================
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# 6. 特征提取器配置
# ==========================================
N_CHANNELS_OUT = 128
IMG_DIM_OUT = 6
P_DIM = IMG_DIM_OUT * IMG_DIM_OUT

# ==========================================
# 7. 硬件配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_config():
    print("=" * 30)
    print(f"MODEL: {MODEL_TYPE} | DATASET: {DATASET_NAME}")
    print(f"ATTENTION: {USE_ATTENTION} | PHI_TH: {PHI_TH:.2e}")
    print("=" * 30)