import torch
import numpy as np
import medmnist

# ==========================================
# 1. 实验控制中心 (Experiment Control)
# ==========================================
# 选择要使用的数据集
# 可选值: 'FASHION_MNIST', 'SVHN', 'BLOOD_MNIST', 'GTSRB', 'MNIST', 'CIFAR10', 'CIFAR100', 'GEOMETRIC_SHAPES', 'MIO_TCD_CLASSIFICATION', 'VEHICLES', 'SHAPES_CLASSIFICATION'
DATASET_NAME = 'MNIST'

# [GTSRB 专属配置] 选择要训练的类别子集
# 如果为 None: 使用全部 43 个类别
# 如果为列表: 仅使用列表中的类别索引 (例如 [0, 1, 2] 只训练前三类)
# 这对于快速验证或专注于特定交通标志非常有用
GTSRB_SUBSET_INDICES = [12,13, 14, 15,17, 33, 34, 35,36,37,38,39,40]  # 示例: Yield, Stop, No vehicles, No Entry, Turn Right, Turn Left, Ahead Only,
# GTSRB_SUBSET_INDICES = None  # 示例: Stop, No Entry, Turn Right, Turn Left, Ahead Only

# [CIFAR100 专属配置] 选择要训练的类别子集
# 如果为 None: 使用全部 100 个类别
# 如果为列表: 仅使用列表中的类别索引 (例如 [0, 1, 2] 只训练前三类)
# 或者使用类别名称列表 (例如 ['apple', 'aquarium_fish', 'baby'])
CIFAR100_SUBSET_INDICES = None  # 使用索引选择，例如 [0, 1, 2, 10, 11, 12]
CIFAR100_SUBSET_NAMES = None    # 使用名称选择，例如 ['apple', 'aquarium_fish', 'baby']

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

# CIFAR100 完整类别名称 (100类)
CIFAR100_ALL_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 动态计算 GTSRB 配置
if GTSRB_SUBSET_INDICES is None:
    _gtsrb_n_classes = 43
    _gtsrb_class_names = GTSRB_ALL_CLASSES
else:
    _gtsrb_n_classes = len(GTSRB_SUBSET_INDICES)
    _gtsrb_class_names = [GTSRB_ALL_CLASSES[i] for i in GTSRB_SUBSET_INDICES]

# 动态计算 CIFAR100 配置
# 首先确定要使用的子集索引
_cifar100_selected_indices = None
_cifar100_n_classes = 100
_cifar100_class_names = CIFAR100_ALL_CLASSES

if DATASET_NAME == 'CIFAR100':
    if CIFAR100_SUBSET_NAMES is not None:
        # 将类别名称转换为索引
        _cifar100_selected_indices = []
        for name in CIFAR100_SUBSET_NAMES:
            if name in CIFAR100_ALL_CLASSES:
                _cifar100_selected_indices.append(CIFAR100_ALL_CLASSES.index(name))
            else:
                raise ValueError(f"未知的 CIFAR100 类别名称: {name}")
    elif CIFAR100_SUBSET_INDICES is not None:
        _cifar100_selected_indices = CIFAR100_SUBSET_INDICES
    
    # 计算类别数和类别名称
    if _cifar100_selected_indices is not None:
        _cifar100_n_classes = len(_cifar100_selected_indices)
        _cifar100_class_names = [CIFAR100_ALL_CLASSES[i] for i in _cifar100_selected_indices]

DATASET_CONFIGS = {
    'FASHION_MNIST': {
        'n_classes': 10, 'in_channels': 1,
        'class_names': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
        'target_size': (28, 28)
    },
    'SVHN': {
        'n_classes': 10, 'in_channels': 3,
        'class_names': [str(i) for i in range(10)],
        'target_size': (32, 32)
    },
    'BLOOD_MNIST': {
        'n_classes': 8, 'in_channels': 3,
        'class_names': list(medmnist.INFO['bloodmnist']['label'].values()),
        'target_size': (32, 32)
    },
    'GTSRB': {
        'n_classes': _gtsrb_n_classes,
        'in_channels': 3,
        'class_names': _gtsrb_class_names,
        # 注意: 为了兼容 models.py 中针对 28x28 输入设计的下采样层，
        # 我们将 GTSRB 统一缩放到 28x28。
        'target_size': (28, 28)
    },
    'MNIST': {
        'n_classes': 10, 'in_channels': 1,
        'class_names': [str(i) for i in range(10)],
        'target_size': (28, 28)
    },
    'CIFAR10': {
        'n_classes': 10, 'in_channels': 3,
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'target_size': (32, 32)
    },
    'CIFAR100': {
        'n_classes': _cifar100_n_classes,
        'in_channels': 3,
        'class_names': _cifar100_class_names,
        'target_size': (32, 32)
    },
    'GEOMETRIC_SHAPES': {
        'n_classes': 8,
        'in_channels': 1,
        'class_names': ['circle', 'ellipse', 'octagon', 'parallelogram', 'pentagon', 'rectangle', 'rhombus', 'square'],
        'target_size': (32, 32)
    },
    'MIO_TCD_CLASSIFICATION': {
        'n_classes': 10,
        'in_channels': 3,
        'class_names': ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle', 'non-motorized_vehicle', 'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van'],
        'target_size': (32, 32)
    },
    'VEHICLES': {
        'n_classes': 7,
        'in_channels': 3,
        'class_names': ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains'],
        'target_size': (64, 64)
    },
    'SHAPES_CLASSIFICATION': {
        'n_classes': 3,
        'in_channels': 1,
        'class_names': ['circles', 'squares', 'triangles'],
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
# 3. 全局训练配置   /*
# ==========================================
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.0005
SEED = 42
DATA_ROOT = './data'

# ==========================================
# 4. DFM-FNCN 模糊模型特定配置
# ==========================================
MAX_RULES = 200
# 针对 GTSRB 这种复杂数据集，建议适当放宽阈值或增大 Sigma
PHI_TH = np.exp(-45)
INIT_SIGMA = 1.05
# ==========================================
# [创新点 1] 结构设计创新：添加通道注意力机制
# ==========================================
USE_ATTENTION = False

# ==========================================
# 5. 传统 CNN 配置
# ==========================================
CNN_CLASSIFIER_NODES = [1024, 512]
CNN_DROPOUT = 0.5

# ==========================================
# 6. 特征提取器输出配置
# ==========================================
N_CHANNELS_OUT =128
if DATASET_NAME == 'VEHICLES':
    IMG_DIM_OUT = 15
elif DATASET_NAME == 'FASHION_MNIST' or DATASET_NAME == 'MNIST' or DATASET_NAME == 'SHAPES_CLASSIFICATION' or DATASET_NAME == 'GTSRB':
    IMG_DIM_OUT = 6
else:
    IMG_DIM_OUT = 7
P_DIM = IMG_DIM_OUT * IMG_DIM_OUT

# ==========================================
# 7. 硬件配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 8. [创新点 2] 学习算法创新 基于聚类的规则初始化
# ==========================================
USE_CLUSTERING_INIT = False  # 是否开启聚类初始化 (初始化后仍可继续动态生成)
N_CLUSTERS = 30           # 初始聚类中心数量
CLUSTERING_SAMPLE_LIMIT = 300000 # 用于聚类的样本数量限制 (避免内存溢出)

# ==========================================
# 9. [创新点 3] 学习算法创新 - 规则修剪
# ==========================================
USE_PRUNING = False          # 是否在训练结束后自动修剪无效规则
# 修剪方法:
# 'CONSEQUENT': 基于后件置信度 (如果规则对任何类别的预测概率都不高，则修剪)
# 'ACTIVATION': 基于激活强度 (如果规则在测试集上的平均激活度极低，则修剪)
PRUNING_METHOD = 'CONSEQUENT'
# 修剪阈值: 
# 对于 'CONSEQUENT': 建议 0.3 ~ 0.5 (最大概率低于此值则修剪)
# 对于 'ACTIVATION': 建议 0.001 ~ 0.01 (平均激活度低于此值则修剪)
PRUNING_THRESHOLD = 0.6


# ==========================================
# 10. [创新点 4] 结构设计创新：注意力引导的解码器
# ==========================================
USE_ATTENTION_GUIDED_DECODER = False  # 是否使用注意力引导的解码器
ATTENTION_GUIDED_DECODER_WEIGHT = 1  # 注意力调制强度 (0.0-1.0)

# ==========================================
# 11. [创新点 5] 结构设计创新：GAN解码器
# ==========================================
USE_GAN_DECODER = False  # 是否使用GAN作为解码器
GAN_ADVERSARIAL_WEIGHT = 0.01  # 对抗损失权重 (相对于重建损失)：
# 如果生成的图像纹理乱七八糟但轮廓对，说明对抗权重太大，减小它。
# 如果生成的图像依然模糊，说明对抗权重太小，增大它。
GAN_DISCRIMINATOR_LR = 0.0002  # 判别器学习率
GAN_GENERATOR_LR = 0.001  # 生成器学习率 (与解码器学习率相同)
GAN_DISCRIMINATOR_UPDATE_RATIO = 1  # 每训练生成器一次，训练判别器的次数
GAN_USE_LSGAN = True  # 使用LSGAN (最小二乘GAN) 损失，否则使用BCE

# ==========================================
# 12. [创新点 6] 多尺度规则可视化(弃用)
# ==========================================
USE_MULTI_SCALE_VISUALIZATION = False  # 是否启用多尺度可视化
MULTI_SCALE_LEVELS = ['coarse', 'medium', 'fine']  # 可视化尺度
MULTI_SCALE_WEIGHTS = [0.3, 0.5, 0.2]  # 各尺度融合权重

# ==========================================
# 12. [创新点 6] 学习方法创新 - 动态规则融合
# ==========================================
USE_RULE_MERGING = False          # 是否启用动态规则融合
# 融合方法:
# 'SIMILARITY': 基于规则中心的相似度 (余弦相似度)
# 'ACTIVATION_CORRELATION': 基于规则激活的相关性
MERGING_METHOD = 'SIMILARITY'
# 融合阈值:
# 对于 'SIMILARITY': 建议 0.8 ~ 0.95 (余弦相似度高于此值则融合)
# 对于 'ACTIVATION_CORRELATION': 建议 0.7 ~ 0.9 (激活相关性高于此值则融合)
MERGING_THRESHOLD = 0.9
# 融合策略:
# 'WEIGHTED_AVERAGE': 加权平均 (根据规则置信度或激活频率)
# 'DOMINANT_RULE': 保留置信度更高的规则
MERGING_STRATEGY = 'WEIGHTED_AVERAGE'
# 融合时机:
# 'EVERY_EPOCH': 每个epoch结束后融合
# 'FINAL_ONLY': 只在训练结束时融合



MERGING_TIMING = 'EVERY_EPOCH'

def print_config():
    print("=" * 30)
    print(f"MODEL: {MODEL_TYPE} | DATASET: {DATASET_NAME}")
    if DATASET_NAME == 'GTSRB' and GTSRB_SUBSET_INDICES is not None:
        print(f"GTSRB Subset: {len(GTSRB_SUBSET_INDICES)} classes selected")
    print(f"ATTENTION: {USE_ATTENTION} | PHI_TH: {PHI_TH:.2e}")
    print(f"CLUSTERING INIT: {USE_CLUSTERING_INIT} (K={N_CLUSTERS})")
    print(f"PRUNING: {USE_PRUNING} (Method: {PRUNING_METHOD}, Th: {PRUNING_THRESHOLD})")
    print(f"ATTENTION GUIDED DECODER: {USE_ATTENTION_GUIDED_DECODER}")
    print(f"RULE MERGING: {USE_RULE_MERGING} (Method: {MERGING_METHOD}, Th: {MERGING_THRESHOLD})")
    print("=" * 30)
