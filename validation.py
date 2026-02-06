import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import pandas as pd
import medmnist
from medmnist import BloodMNIST

import config as cfg
from models import FullModel, TraditionalCNNModel, PlantNetANFIS, PlantNetSimple

# 移除硬编码的 RUN_DIR_TO_VALIDATE

def configure_model_from_checkpoint(checkpoint):
    """从检查点中的 'config_params' 推断配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法自动推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")


def load_model_and_config(model_path):
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        sys.exit()

    print(f"正在从 {model_path} 加载模型...")
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE,weights_only=False)

    configure_model_from_checkpoint(checkpoint)

    if cfg.MODEL_TYPE == 'DFM_FNCN':
        if 'max_rules' not in checkpoint:
            print("错误: 模糊模型检查点不包含 'max_rules'。")
            sys.exit()
        cfg.MAX_RULES = checkpoint['max_rules']
        # [创新点1] 恢复 Attention 配置
        if 'config_params' in checkpoint and 'USE_ATTENTION' in checkpoint['config_params']:
            cfg.USE_ATTENTION = checkpoint['config_params']['USE_ATTENTION']

        print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}, USE_ATTENTION = {cfg.USE_ATTENTION}")

    if cfg.MODEL_TYPE == 'DFM_FNCN':
        model = FullModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'TRADITIONAL_CNN':
        model = TraditionalCNNModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'PLANTNET_ANFIS':
        model = PlantNetANFIS(num_classes=cfg.N_CLASSES, use_fuzzy_layer=True, in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'PLANTNET_SIMPLE':
        model = PlantNetSimple(num_classes=cfg.N_CLASSES, in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
    else:
        raise ValueError(f"未知的 MODEL_TYPE: {cfg.MODEL_TYPE}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_test_loader():
    if cfg.IN_CHANNELS == 1:
        norm_mean, norm_std = (0.5,), (0.5,)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # 构建转换列表
    transform_list = [transforms.Resize(cfg.TARGET_SIZE)]
    # 对于 GEOMETRIC_SHAPES 数据集，添加灰度转换
    if cfg.DATASET_NAME == 'GEOMETRIC_SHAPES' or cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION':
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    data_transform = transforms.Compose(transform_list)

    if cfg.DATASET_NAME == 'FASHION_MNIST':
        test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'SVHN':
        test_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        test_dataset = BloodMNIST(split='test', transform=data_transform, download=True, root=cfg.DATA_ROOT)
    elif cfg.DATASET_NAME == 'GTSRB':
        # [修改] GTSRB 子集处理逻辑
        target_transform = None
        if cfg.GTSRB_SUBSET_INDICES is not None:
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(cfg.GTSRB_SUBSET_INDICES)}
            target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))

        test_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='test', download=True,
                                      transform=data_transform, target_transform=target_transform)

        if cfg.GTSRB_SUBSET_INDICES is not None:
            subset_set = set(cfg.GTSRB_SUBSET_INDICES)
            test_indices = [i for i, (_, label) in enumerate(test_dataset._samples) if label in subset_set]
            test_dataset = Subset(test_dataset, test_indices)
    
    elif cfg.DATASET_NAME == 'MNIST':
        test_dataset = datasets.MNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
    
    elif cfg.DATASET_NAME == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
    
    elif cfg.DATASET_NAME == 'CIFAR100':
        # CIFAR100 子集处理逻辑
        target_transform = None
        
        # 确定要使用的子集索引
        selected_indices = None
        if cfg.CIFAR100_SUBSET_NAMES is not None:
            # 将类别名称转换为索引
            selected_indices = []
            for name in cfg.CIFAR100_SUBSET_NAMES:
                if name in cfg.CIFAR100_ALL_CLASSES:
                    selected_indices.append(cfg.CIFAR100_ALL_CLASSES.index(name))
                else:
                    raise ValueError(f"未知的 CIFAR100 类别名称: {name}")
        elif cfg.CIFAR100_SUBSET_INDICES is not None:
            selected_indices = cfg.CIFAR100_SUBSET_INDICES
        
        test_dataset = datasets.CIFAR100(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
        
        if selected_indices is not None:
            # 1. 创建标签映射: 原始ID -> 0..N-1
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
            # 2. 过滤数据集
            subset_set = set(selected_indices)
            test_indices = []
            
            # 测试集过滤
            test_targets = test_dataset.targets if hasattr(test_dataset, 'targets') else test_dataset.targets
            for i, label in enumerate(test_targets):
                if label in subset_set:
                    test_indices.append(i)
            
            test_dataset = Subset(test_dataset, test_indices)
            print(f"CIFAR100 Subset: Test {len(test_dataset)}")
    
    elif cfg.DATASET_NAME == 'GEOMETRIC_SHAPES':
        # 加载整个数据集
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'geometric_shapes'), transform=data_transform)
        # 使用与训练相同的随机种子进行分割 (80% 训练, 20% 测试)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Geometric Shapes 测试集大小: {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'MIO_TCD_CLASSIFICATION':
        # 加载整个数据集
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'MIO-TCD-Classification'), transform=data_transform)
        # 使用与训练相同的随机种子进行分割 (训练集占5/6，测试集占1/6)
        train_ratio = 5/6
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"MIO-TCD-Classification 测试集大小: {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'VEHICLES':
        # 加载整个数据集
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'Vehicles'), transform=data_transform)
        # 使用与训练相同的随机种子进行分割 (训练集占4/5，测试集占1/5)
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Vehicles 测试集大小: {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION':
        # 加载 Shapes Classification 数据集
        # 路径: data/Shapes_Classification/archive(6)/shapes/
        dataset_path = os.path.join(cfg.DATA_ROOT, 'Shapes_Classification', 'archive(6)', 'shapes')
        full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
        # 使用与训练相同的随机种子进行分割 (训练集占4/5，测试集占1/5)
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Shapes Classification 测试集大小: {len(test_dataset)}")
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    return test_loader


def run_inference(model, loader):
    print("正在测试集上运行推理...")
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data_tuple in loader:
            data, targets = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            if cfg.MODEL_TYPE == 'DFM_FNCN':
                outputs = model(data, labels=None, training_phase=False)
            else:
                outputs = model(data)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    print("正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16)
    plt.ylabel('真实类别', fontsize=12)
    plt.xlabel('预测类别', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵已保存至: {output_path}")
    plt.close()


def run_validation(run_dir):
    """[修改] 接收 run_dir 参数供 main.py 调用"""
    print(f"\n>>> 开始评估 (Validation): {run_dir}")
    model_path = os.path.join(run_dir, 'best_model.pth')
    output_dir = run_dir

    model = load_model_and_config(model_path)
    test_loader = get_test_loader()
    predictions, targets = run_inference(model, test_loader)

    print("\n" + "=" * 50)
    print("           分类报告 (Classification Report)")
    print("=" * 50)
    report_str = classification_report(targets, predictions, target_names=cfg.CLASS_NAMES, zero_division=0)
    print(report_str)
    print("=" * 50 + "\n")

    report_dict = classification_report(targets, predictions, target_names=cfg.CLASS_NAMES, zero_division=0, output_dict=True)
    accuracy_value = report_dict.pop('accuracy') # type: ignore
    df = pd.DataFrame(report_dict).transpose()
    df.loc['accuracy'] = pd.Series({'f1-score': accuracy_value, 'support': df.loc['weighted avg', 'support']})
    csv_save_path = os.path.join(output_dir, 'classification_report.csv')
    df.to_csv(csv_save_path, index=True, encoding='utf-8-sig')

    plot_confusion_matrix(targets, predictions, cfg.CLASS_NAMES, output_dir)
    print(f"评估完成。结果已保存到 {output_dir}")

if __name__ == '__main__':
    # 仅用于单独测试，需手动指定路径
    TEST_DIR = 'checkpoints/VEHICLES_DFM_FNCN_RESNET18_PRETRAINED_20260102_185802'
    if os.path.exists(TEST_DIR):
        run_validation(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")
