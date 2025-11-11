import torch
import numpy as np

# ==========================================
# 实验控制中心
# ==========================================
MODEL_TYPE = 'DFM_FNCN'
EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'

# ==========================================
# 全局训练配置
# ==========================================
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
SEED = 42

# ==========================================
# 数据集配置
# ==========================================
N_CLASSES = 10
DATA_ROOT = './data'

# ==========================================
# DFM-FNCN 模糊模型特定配置
# ==========================================
MAX_RULES = 50
PHI_TH = np.exp(-24)
INIT_SIGMA = 1.5

# ==========================================
# 传统 CNN 分类器配置
# ==========================================
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# [关键修改] 特征提取器输出配置
# ==========================================
N_CHANNELS = 128  # 从 32 更改为 128
IMG_DIM = 6  # 从 7 更改为 6
P_DIM = IMG_DIM * IMG_DIM  # 自动更新为 36

# ==========================================
# 硬件配置
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

    # [新] 打印特征图形状
    print(f"FEATURE_MAP_SHAPE: ({N_CHANNELS}, {IMG_DIM}, {IMG_DIM})")

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