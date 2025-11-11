import torch
import numpy as np

# ==========================================
# [新] 实验控制中心
# ==========================================
# 选择要训练的模型:
# 'DFM_FNCN'          - 论文复现的模糊分类器
# 'TRADITIONAL_CNN'   - 传统的全连接层分类器 (用于对比)
MODEL_TYPE = 'DFM_FNCN'

# 选择要使用的特征提取器 (编码器):
# 'RESNET18_PRETRAINED' - 强大的预训练 ResNet18
# 'VGG16_PRETRAINED'    - [新] 强大的预训练 VGG16
# 'SIMPLE_CNN'          - 一个简单的轻量级 CNN
EXTRACTOR_TYPE = 'VGG16_PRETRAINED'

# ==========================================
# 全局训练配置
# ==========================================
# ... (保持不变) ...
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.005
SEED = 42

# ==========================================
# 数据集配置
# ==========================================
# ... (保持不变) ...
N_CLASSES = 10
DATA_ROOT = './data'

# ==========================================
# DFM-FNCN 模糊模型特定配置
# ==========================================
# ... (保持不变) ...
MAX_RULES = 50
PHI_TH = np.exp(-16)
INIT_SIGMA = 1.05

# ==========================================
# 传统 CNN 分类器配置
# ==========================================
# ... (保持不变) ...
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# 特征提取器输出配置 (勿动)
# ==========================================
# ... (保持不变) ...
N_CHANNELS = 32
IMG_DIM = 7
P_DIM = IMG_DIM * IMG_DIM # 49

# ==========================================
# 硬件配置
# ==========================================
# ... (保持不变) ...
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_config():
    print("=" * 30)
    print("实验配置 (Config)")
    print("=" * 30)
    print(f"DEVICE: {DEVICE}")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"EXTRACTOR_TYPE: {EXTRACTOR_TYPE}")

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