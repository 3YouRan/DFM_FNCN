from math import exp

import torch

# ==========================================
# 全局训练配置
# ==========================================
BATCH_SIZE = 64
EPOCHS = 10             # 建议至少跑 10 个 epoch 以观察规则增长
LR = 0.002              # 学习率
SEED = 42               # 随机种子，保证结果可复现

# ==========================================
# 数据集配置 (Fashion-MNIST)
# ==========================================
N_CLASSES = 10
DATA_ROOT = './data'

# ==========================================
# DFM-FNCN 模型特定配置
# ==========================================
# 1. DCNN 特征提取器输出设置
N_CHANNELS = 32         # 特征图数量 (对应论文中的 N)
IMG_DIM = 7             # 特征图空间尺寸 (7x7)
P_DIM = IMG_DIM * IMG_DIM # 展平后的特征维度 (p = w * h = 49)

# 2. 动态规则生成超参数
MAX_RULES = 50         # 规则池的最大容量
PHI_TH = exp(-19.5)           # [关键阈值] 激发强度阈值。低于此值将触发新规则生成。
                        # 调大 -> 生成更多规则；调小 -> 生成更少规则。
INIT_SIGMA = 1.2        # 新生成规则的初始宽度 (Sigma)

# ==========================================
# 硬件配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_config():
    print("="*30)
    print("实验配置 (Config)")
    print("="*30)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    print(f"DFM-FNCN: Channels={N_CHANNELS}, Feature Dim={IMG_DIM}x{IMG_DIM} (p={P_DIM})")
    print(f"Dynamic Rules: Max={MAX_RULES}, Threshold(phi_th)={PHI_TH}, Init Sigma={INIT_SIGMA}")
    print("="*30)