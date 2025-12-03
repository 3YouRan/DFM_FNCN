import torch
import numpy as np
import medmnist

# ==========================================
# 1. 实验控制中心 (Experiment Control)
# ==========================================
# 选择要使用的数据集
# 可选值: 'FASHION_MNIST', 'SVHN', 'BLOOD_MNIST', 'GTSRB'
DATASET_NAME = 'GTSRB'

# [GTSRB 专属配置] 选择要训练的类别子集
# 如果为 None: 使用全部 43 个类别
# 如果为列表: 仅使用列表中的类别索引 (例如 [0, 1, 2] 只训练前三类)
# 这对于快速验证或专注于特定交通标志非常有用
GTSRB_SUBSET_INDICES = [13, 14, 15, 17, 33, 34, 35]  # 示例: Yield, Stop, No vehicles, No Entry, Turn Right, Turn Left, Ahead Only,
# GTSRB_SUBSET_INDICES = None  # 示例: Stop, No Entry, Turn Right, Turn Left, Ahead Only

# 选择模型架构
# 'DFM_FNCN': 论文复现的模糊神经网络
# 'TRADITIONAL_CNN': 传统的深度卷积神经网络
MODEL_TYPE = 'DFM_FNCN'

# 选择特征提取器
# 'RESNET18_PRETRAINED', 'VGG16_PRETRAINED', 'SIMPLE_CNN'
EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'

# ==========================================
# 2. 数据集配置 (Dataset Configuration)
# ==========================================

# GTSRB 完整类别名称 (43类)
GTSRB_ALL_CLASSES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited',
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End no passing veh > 3.5 tons'
]

# 动态计算 GTSRB 配置
if GTSRB_SUBSET_INDICES is None:
    _gtsrb_n_classes = 43
    _gtsrb_class_names = GTSRB_ALL_CLASSES
else:
    _gtsrb_n_classes = len(GTSRB_SUBSET_INDICES)
    _gtsrb_class_names = [GTSRB_ALL_CLASSES[i] for i in GTSRB_SUBSET_INDICES]

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
    },
    'GTSRB': {
        'n_classes': _gtsrb_n_classes,
        'in_channels': 3,
        'class_names': _gtsrb_class_names,
        # 注意: 为了兼容 models.py 中针对 28x28 输入设计的下采样层，
        # 我们将 GTSRB 统一缩放到 28x28。
        'target_size': (28, 28)
    }
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"未知的 DATASET_NAME: {DATASET_NAME}")

# 自动加载当前数据集参数
CURRENT_DATA_CONFIG = DATASET_CONFIGS[DATASET_NAME]
N_CLASSES = CURRENT_DATA_CONFIG['n_classes']
IN_CHANNELS = CURRENT_DATA_CONFIG['in_channels']
CLASS_NAMES = CURRENT_DATA_CONFIG['class_names']
TARGET_SIZE = CURRENT_DATA_CONFIG['target_size']

# ==========================================
# 3. 全局训练配置
# ==========================================
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0005
SEED = 42
DATA_ROOT = './data'

# ==========================================
# 4. DFM-FNCN 模糊模型特定配置
# ==========================================
MAX_RULES = 100
# 针对 GTSRB 这种复杂数据集，建议适当放宽阈值或增大 Sigma
PHI_TH = np.exp(-59)
INIT_SIGMA = 1
USE_ATTENTION = True

# ==========================================
# 5. 传统 CNN 配置
# ==========================================
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# 6. 特征提取器输出配置
# ==========================================
N_CHANNELS_OUT = 128
IMG_DIM_OUT = 6
P_DIM = IMG_DIM_OUT * IMG_DIM_OUT

# ==========================================
# 7. 硬件配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 8. [创新点 3] 基于聚类的规则初始化
# ==========================================
USE_CLUSTERING_INIT = True  # 是否开启聚类初始化 (初始化后仍可继续动态生成)
N_CLUSTERS = 7             # 初始聚类中心数量 (即初始规则数，应当与分类数一致)
CLUSTERING_SAMPLE_LIMIT = 10000 # 用于聚类的样本数量限制 (避免内存溢出)

# ==========================================
# 9. [创新点 2] 结构学习 - 规则修剪
# ==========================================
USE_PRUNING = True          # 是否在训练结束后自动修剪无效规则
# 修剪方法:
# 'CONSEQUENT': 基于后件置信度 (如果规则对任何类别的预测概率都不高，则修剪)
# 'ACTIVATION': 基于激活强度 (如果规则在测试集上的平均激活度极低，则修剪)
PRUNING_METHOD = 'CONSEQUENT'
# 修剪阈值:
# 对于 'CONSEQUENT': 建议 0.3 ~ 0.5 (最大概率低于此值则修剪)
# 对于 'ACTIVATION': 建议 0.001 ~ 0.01 (平均激活度低于此值则修剪)
PRUNING_THRESHOLD = 0.6

def print_config():
    print("=" * 30)
    print(f"MODEL: {MODEL_TYPE} | DATASET: {DATASET_NAME}")
    if DATASET_NAME == 'GTSRB' and GTSRB_SUBSET_INDICES is not None:
        print(f"GTSRB Subset: {len(GTSRB_SUBSET_INDICES)} classes selected")
    print(f"ATTENTION: {USE_ATTENTION} | PHI_TH: {PHI_TH:.2e}")
    print(f"CLUSTERING INIT: {USE_CLUSTERING_INIT} (K={N_CLUSTERS})")
    print(f"PRUNING: {USE_PRUNING} (Method: {PRUNING_METHOD}, Th: {PRUNING_THRESHOLD})")
    print("=" * 30)