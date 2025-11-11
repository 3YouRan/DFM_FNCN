import torch
import numpy as np
import medmnist  # 确保已 pip install medmnist

# ==========================================
# 1. 实验控制中心 (在此处配置您的运行)
# ==========================================

# 1.1 选择数据集:
# 'FASHION_MNIST' - 10类, 1通道, 28x28
# 'SVHN'            - 10类, 3通道, 32x32 (将缩放至 28x28)
# 'BLOOD_MNIST'     - 8类,  3通道, 28x28 (来自 MedMNIST)
DATASET_NAME = 'SVHN'

# 1.2 选择要训练的模型:
# 'DFM_FNCN'          - 论文复现的模糊分类器
# 'TRADITIONAL_CNN'   - 传统的全连接层分类器 (用于对比)
MODEL_TYPE = 'DFM_FNCN'

# 1.3 选择要使用的特征提取器 (编码器):
# 'RESNET18_PRETRAINED' - 强大的预训练 ResNet18
# 'VGG16_PRETRAINED'    - 强大的预训练 VGG16
# 'SIMPLE_CNN'          - 一个简单的轻量级 CNN
EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'

# ==========================================
# 2. 数据集配置 (自动从 DATASET_NAME 加载)
# ==========================================
DATASET_CONFIGS = {
    'FASHION_MNIST': {
        'n_classes': 10,
        'in_channels': 1,
        'class_names': [
            'T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ],
        'target_size': (28, 28)
    },
    'SVHN': {
        'n_classes': 10,
        'in_channels': 3,
        'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'target_size': (28, 28)  # SVHN 是 32x32，我们将缩放它
    },
    'BLOOD_MNIST': {
        'n_classes': 8,
        'in_channels': 3,
        'class_names': list(medmnist.INFO['bloodmnist']['label'].values()),
        'target_size': (28, 28)
    }
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"未知的 DATASET_NAME: {DATASET_NAME}")

# --- 自动加载当前配置 ---
CURRENT_DATA_CONFIG = DATASET_CONFIGS[DATASET_NAME]
N_CLASSES = CURRENT_DATA_CONFIG['n_classes']
IN_CHANNELS = CURRENT_DATA_CONFIG['in_channels']
CLASS_NAMES = CURRENT_DATA_CONFIG['class_names']
TARGET_SIZE = CURRENT_DATA_CONFIG['target_size']
# ---

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
PHI_TH = np.exp(-56)
INIT_SIGMA = 1.34

# ==========================================
# 5. 传统 CNN 分类器配置
# ==========================================
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# 6. 特征提取器输出配置 (勿动)
# ==========================================
# 我们的架构将始终将 (28, 28) 缩减到 (7, 7)
# 然后 VGG/ResNet 将其缩减到 (6, 6)
# 为保持一致性，所有提取器都必须输出 6x6
N_CHANNELS_OUT = 128  # 提取器的输出通道
IMG_DIM_OUT = 6  # 提取器的输出空间维度
P_DIM = IMG_DIM_OUT * IMG_DIM_OUT  # 36

# ==========================================
# 7. 硬件配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_config():
    """打印当前配置"""
    print("=" * 30)
    print("实验配置 (Config)")
    print("=" * 30)
    print(f"DEVICE: {DEVICE}")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"EXTRACTOR_TYPE: {EXTRACTOR_TYPE}")
    print(f"DATASET: {DATASET_NAME} ({N_CLASSES} 类, {IN_CHANNELS} 通道)")

    # 打印特征图形状
    print(f"FEATURE_MAP_SHAPE: ({N_CHANNELS_OUT}, {IMG_DIM_OUT}, {IMG_DIM_OUT})")

    if MODEL_TYPE == 'DFM_FNCN':
        print("\n--- DFM-FNCN 配置 ---")
        print(f"Phi Threshold: {PHI_TH:.2e}")
        print(f"Init Sigma: {INIT_SIGMA}")
        print(f"Max Rules: {MAX_RULES}")
    else:
        print("\n--- Traditional CNN 配置 ---")
        print(f"Classifier Layers: {CNN_CLASSIFIER_NODES}")
        print(f"Dropout: {CNN_DROPOUT}")
    print("=" * 30)