import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from datetime import datetime
import time
from tqdm import tqdm
import medmnist
from medmnist import BloodMNIST
# [修改] 更新 AMP 导入
from torch.amp import autocast,GradScaler # type: ignore

import scipy.special
# [创新点 3] 导入 MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
import logging  # 新增：导入日志模块

import config as cfg
from models import FullModel, TraditionalCNNModel

CLASS_NAMES = cfg.CLASS_NAMES

def setup_logger(save_path):
    """设置训练日志记录器"""
    log_file = os.path.join(save_path, 'training_log.txt')

    # 创建日志记录器
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # 清除已有的处理器（避免重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_data_loaders():
    """
    加载数据并计算类别权重以解决样本不平衡问题。
    返回: train_loader, test_loader, class_weights
    """
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

    # 初始化类别计数
    class_counts = torch.zeros(cfg.N_CLASSES)

    if cfg.DATASET_NAME == 'FASHION_MNIST':
        train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
        # 统计样本
        labels = train_dataset.targets
        class_counts = torch.bincount(labels, minlength=cfg.N_CLASSES).float()

    elif cfg.DATASET_NAME == 'SVHN':
        train_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform)
        test_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform)
        # 统计样本
        labels = torch.tensor(train_dataset.labels)
        class_counts = torch.bincount(labels, minlength=cfg.N_CLASSES).float()

    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        train_dataset = BloodMNIST(split='train', transform=data_transform, download=True, root=cfg.DATA_ROOT)
        test_dataset = BloodMNIST(split='test', transform=data_transform, download=True, root=cfg.DATA_ROOT)
        # 统计样本
        labels = torch.tensor(train_dataset.labels.squeeze())
        class_counts = torch.bincount(labels, minlength=cfg.N_CLASSES).float()

    elif cfg.DATASET_NAME == 'GTSRB':
        # [修改] GTSRB 子集处理逻辑 + 样本统计
        target_transform = None

        # 加载原始数据集
        train_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform)
        test_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform)

        if cfg.GTSRB_SUBSET_INDICES is not None:
            # 1. 创建标签映射: 原始ID -> 0..N-1
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(cfg.GTSRB_SUBSET_INDICES)}
            # 设置 target_transform
            target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))

            # 应用 transform 到 dataset (注意：Subset 不会自动应用 transform 到内部数据，需手动处理或依赖 dataset 的 transform)
            # torchvision 的 GTSRB 支持 target_transform
            train_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='train', download=True,
                                           transform=data_transform, target_transform=target_transform)
            test_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='test', download=True,
                                          transform=data_transform, target_transform=target_transform)

            # 2. 过滤数据集并统计样本
            subset_set = set(cfg.GTSRB_SUBSET_INDICES)
            train_indices = []

            # 遍历原始样本进行筛选和统计
            print("正在筛选 GTSRB 子集并统计类别分布...")
            for i, (_, label) in enumerate(train_dataset._samples):
                if label in subset_set:
                    train_indices.append(i)
                    # 映射后的标签
                    mapped_label = mapping[label]
                    class_counts[mapped_label] += 1

            test_indices = [i for i, (_, label) in enumerate(test_dataset._samples) if label in subset_set]

            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            print(f"GTSRB Subset: Train {len(train_dataset)}, Test {len(test_dataset)}")
        else:
            # 使用全部数据
            for _, label in train_dataset._samples:
                class_counts[label] += 1

    elif cfg.DATASET_NAME == 'MNIST':
        train_dataset = datasets.MNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        test_dataset = datasets.MNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
        # 统计样本
        labels = train_dataset.targets
        class_counts = torch.bincount(labels, minlength=cfg.N_CLASSES).float()

    elif cfg.DATASET_NAME == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR10(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
        # 统计样本
        labels = torch.tensor(train_dataset.targets)
        class_counts = torch.bincount(labels, minlength=cfg.N_CLASSES).float()

    elif cfg.DATASET_NAME == 'CIFAR100':
        # CIFAR100 子集处理逻辑 (参考 GTSRB)
        target_transform = None
        
        # 加载原始数据集
        train_dataset = datasets.CIFAR100(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR100(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
        
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
        
        if selected_indices is not None:
            # 1. 创建标签映射: 原始ID -> 0..N-1
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
            # 设置 target_transform
            target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))
            
            # 2. 过滤数据集并统计样本
            subset_set = set(selected_indices)
            train_indices = []
            test_indices = []
            
            print("正在筛选 CIFAR100 子集并统计类别分布...")
            
            # 训练集过滤 - 使用 targets 属性
            train_targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.targets
            for i, label in enumerate(train_targets):
                if label in subset_set:
                    train_indices.append(i)
                    # 映射后的标签
                    mapped_label = mapping[label]
                    class_counts[mapped_label] += 1
            
            # 测试集过滤
            test_targets = test_dataset.targets if hasattr(test_dataset, 'targets') else test_dataset.targets
            for i, label in enumerate(test_targets):
                if label in subset_set:
                    test_indices.append(i)
            
            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            print(f"CIFAR100 Subset: Train {len(train_dataset)}, Test {len(test_dataset)}")
        else:
            # 使用全部数据
            train_targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.targets
            for label in train_targets:
                class_counts[label] += 1

    elif cfg.DATASET_NAME == 'GEOMETRIC_SHAPES':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'geometric_shapes'), transform=data_transform)
        # 统计类别样本
        class_counts = torch.zeros(cfg.N_CLASSES)
        for _, label in full_dataset.samples:
            class_counts[label] += 1
        # 按 80% 训练, 20% 测试分割
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        # 需要从子集中获取标签以重新计算类别计数（仅训练集）
        # 由于 random_split 不保留标签属性，我们通过遍历来统计
        class_counts = torch.zeros(cfg.N_CLASSES)
        for idx in train_dataset.indices:
            _, label = full_dataset.samples[idx]
            class_counts[label] += 1
        print(f"Geometric Shapes 数据集: 总共 {len(full_dataset)} 张图像, 训练 {len(train_dataset)}, 测试 {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'MIO_TCD_CLASSIFICATION':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'MIO-TCD-Classification'), transform=data_transform)
        # 统计类别样本
        class_counts = torch.zeros(cfg.N_CLASSES)
        for _, label in full_dataset.samples:
            class_counts[label] += 1
        # 按 5:1 分割 (训练集:测试集 = 5:1) 即训练集占5/6，测试集占1/6
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        # 重新计算训练集的类别计数
        class_counts = torch.zeros(cfg.N_CLASSES)
        for idx in train_dataset.indices:
            _, label = full_dataset.samples[idx]
            class_counts[label] += 1
        print(f"MIO-TCD-Classification 数据集: 总共 {len(full_dataset)} 张图像, 训练 {len(train_dataset)}, 测试 {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'VEHICLES':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'Vehicles'), transform=data_transform)
        # 统计类别样本
        class_counts = torch.zeros(cfg.N_CLASSES)
        for _, label in full_dataset.samples:
            class_counts[label] += 1
        # 按 4:1 分割 (训练集:测试集 = 4:1) 即训练集占80%，测试集占20%
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        # 重新计算训练集的类别计数
        class_counts = torch.zeros(cfg.N_CLASSES)
        for idx in train_dataset.indices:
            _, label = full_dataset.samples[idx]
            class_counts[label] += 1
        print(f"Vehicles 数据集: 总共 {len(full_dataset)} 张图像, 训练 {len(train_dataset)}, 测试 {len(test_dataset)}")
    elif cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION':
        # 加载 Shapes Classification 数据集
        # 路径: data/Shapes_Classification/archive(6)/shapes/
        dataset_path = os.path.join(cfg.DATA_ROOT, 'Shapes_Classification', 'archive(6)', 'shapes')
        full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
        # 统计类别样本
        class_counts = torch.zeros(cfg.N_CLASSES)
        for _, label in full_dataset.samples:
            class_counts[label] += 1
        # 按 8:2 分割 (训练集:测试集 = 8:2)
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        # 重新计算训练集的类别计数
        class_counts = torch.zeros(cfg.N_CLASSES)
        for idx in train_dataset.indices:
            _, label = full_dataset.samples[idx]
            class_counts[label] += 1
        print(f"Shapes Classification 数据集: 总共 {len(full_dataset)} 张图像, 训练 {len(train_dataset)}, 测试 {len(test_dataset)}")
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    # [关键步骤] 计算类别权重 (Inverse Class Frequency)
    # 权重 = 总样本数 / (类别数 * 该类样本数)
    # 或者简单地: 1 / count
    print(f"类别样本统计: {class_counts.tolist()}")

    # 避免除以零
    class_counts = class_counts + 1e-6
    weights = 1.0 / class_counts
    # 归一化权重，使其均值为 1，保持 loss 的量级
    weights = weights / weights.sum() * cfg.N_CLASSES

    print(f"计算得到的类别权重: {weights.tolist()}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    print(f"成功加载 {cfg.DATASET_NAME} 数据集。")

    return train_loader, test_loader, weights

def perform_clustering_initialization(model, train_loader, logger=None):
    """[创新点 3] 使用 MiniBatchKMeans 聚类初始化规则中心"""
    log_msg = f"\n>>> 正在执行聚类初始化 (MiniBatchKMeans, K={cfg.N_CLUSTERS})..."
    print(log_msg)
    if logger:
        logger.info(log_msg)

    model.eval()
    features_list = []
    labels_list = []

    # 1. 收集特征
    log_msg = "正在收集样本特征..."
    print(log_msg)
    if logger:
        logger.info(log_msg)

    with torch.no_grad():
        for i, data_tuple in enumerate(train_loader):
            if i * cfg.BATCH_SIZE > cfg.CLUSTERING_SAMPLE_LIMIT:
                break
            data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1]

            # Extract features: (B, C, H, W)
            feats = model.extractor(data)
            # Flatten for clustering: (B, C * H * W)
            b = feats.size(0)
            feats_flat = feats.view(b, -1).cpu().numpy()

            features_list.append(feats_flat)
            labels_list.append(target.numpy())

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0).flatten()

    log_msg = f"收集了 {all_features.shape[0]} 个样本用于聚类。"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    # 2. 执行聚类
    log_msg = "正在运行 MiniBatchKMeans..."
    print(log_msg)
    if logger:
        logger.info(log_msg)

    kmeans = MiniBatchKMeans(n_clusters=cfg.N_CLUSTERS, n_init=10, random_state=cfg.SEED, batch_size=1024)
    cluster_labels = kmeans.fit_predict(all_features)
    cluster_centers = kmeans.cluster_centers_  # (K, Feature_Dim)

    # 3. 计算每个聚类的多数类标签 (用于初始化后件)
    cluster_majority_classes = []
    for k in range(cfg.N_CLUSTERS):
        # 找到属于该聚类的样本索引
        indices = np.where(cluster_labels == k)[0]
        if len(indices) > 0:
            # 找到这些样本对应的真实标签
            k_labels = all_labels[indices]
            # 确保 k_labels 是一维数组
            k_labels = k_labels.flatten()
            # 多数投票
            counts = np.bincount(k_labels, minlength=cfg.N_CLASSES)
            majority_class = np.argmax(counts)
            cluster_majority_classes.append(majority_class)
        else:
            cluster_majority_classes.append(0)  # Fallback

    # 4. 将中心和类别传回模型
    cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32).to(cfg.DEVICE)
    cluster_centers_reshaped = cluster_centers_tensor.view(cfg.N_CLUSTERS, cfg.N_CHANNELS_OUT, cfg.P_DIM)

    model.classifier.init_rules_from_cluster_centers(cluster_centers_reshaped, cluster_majority_classes)

    log_msg = "聚类初始化完成。\n"
    print(log_msg)
    if logger:
        logger.info(log_msg)

def perform_rule_pruning(model, test_loader, logger=None):
    """[创新点 2] 执行规则修剪"""
    log_msg = f"\n>>> 正在执行规则修剪 (Method: {cfg.PRUNING_METHOD}, Th: {cfg.PRUNING_THRESHOLD})..."
    print(log_msg)
    if logger:
        logger.info(log_msg)

    model.eval()
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    if num_rules == 0:
        log_msg = "没有激活的规则，跳过修剪。"
        print(log_msg)
        if logger:
            logger.info(log_msg)
        return

    keep_indices = []

    if cfg.PRUNING_METHOD == 'CONSEQUENT':
        consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
        consequents_prob = scipy.special.softmax(consequents, axis=1)
        max_probs = np.max(consequents_prob, axis=1)

        for i in range(num_rules):
            if max_probs[i] >= cfg.PRUNING_THRESHOLD:
                keep_indices.append(i)
            else:
                prune_msg = f"    [Prune] Rule {i} dropped (Max Prob: {max_probs[i]:.4f} < {cfg.PRUNING_THRESHOLD})"
                print(prune_msg)
                if logger:
                    logger.info(prune_msg)

    elif cfg.PRUNING_METHOD == 'ACTIVATION':
        log_msg = "正在计算测试集上的规则激活度..."
        print(log_msg)
        if logger:
            logger.info(log_msg)

        total_activations = torch.zeros(num_rules, device=cfg.DEVICE)
        total_samples = 0

        with torch.no_grad():
            for data_tuple in test_loader:
                data = data_tuple[0].to(cfg.DEVICE)
                features = model.extractor(data)
                phi = classifier.get_rule_activations(features)
                if phi is not None:
                    total_activations += torch.sum(phi, dim=0)
                    total_samples += data.size(0)

        avg_activations = (total_activations / total_samples).cpu().numpy()

        for i in range(num_rules):
            if avg_activations[i] >= cfg.PRUNING_THRESHOLD:
                keep_indices.append(i)
            else:
                prune_msg = f"    [Prune] Rule {i} dropped (Avg Act: {avg_activations[i]:.5f} < {cfg.PRUNING_THRESHOLD})"
                print(prune_msg)
                if logger:
                    logger.info(prune_msg)

    else:
        log_msg = f"未知的修剪方法: {cfg.PRUNING_METHOD}"
        print(log_msg)
        if logger:
            logger.warning(log_msg)
        return

    if len(keep_indices) < num_rules:
        classifier.prune_rules(keep_indices)
        log_msg = f"规则修剪完成，从 {num_rules} 条规则修剪到 {len(keep_indices)} 条。\n"
    else:
        log_msg = "没有规则被修剪。\n"

    print(log_msg)
    if logger:
        logger.info(log_msg)

def perform_rule_merging(model, test_loader, epoch=None, logger=None):
    """[创新点 6] 执行规则融合"""
    if not cfg.USE_RULE_MERGING:
        return 0, []

    # 检查融合时机
    if cfg.MERGING_TIMING == 'EVERY_EPOCH' and epoch is not None:
        # 每个epoch结束后融合
        log_msg = f"\n>>> Epoch {epoch+1} 结束后执行规则融合..."
        print(log_msg)
        if logger:
            logger.info(log_msg)

        merged_count, merge_pairs = model.classifier.merge_similar_rules(test_loader)
        return merged_count, merge_pairs
    elif cfg.MERGING_TIMING == 'FINAL_ONLY' and epoch is None:
        # 只在训练结束时融合
        log_msg = f"\n>>> 训练结束后执行规则融合..."
        print(log_msg)
        if logger:
            logger.info(log_msg)

        merged_count, merge_pairs = model.classifier.merge_similar_rules(test_loader)
        return merged_count, merge_pairs
    else:
        return 0, []

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler, logger=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # 使用单行动态刷新的进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch+1}", ncols=100, leave=False, 
                smoothing=1, bar_format='{l_bar}{bar}{r_bar}')

    for batch_idx, data_tuple in pbar:
        data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
        if target.ndim == 2 and target.shape[1] == 1: target = target.squeeze(1)

        optimizer.zero_grad()
        with autocast(device_type=cfg.DEVICE.type, enabled=(cfg.DEVICE.type == 'cuda')):
            if cfg.MODEL_TYPE == 'DFM_FNCN':
                output = model(data, labels=target, training_phase=True)
            else:
                output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            model.classifier.commit_pending_rule()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 动态更新进度条信息
        acc = 100. * correct / total
        if cfg.MODEL_TYPE == 'DFM_FNCN':
            rules = model.classifier.num_active_rules.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'rules': f'{rules}/{cfg.MAX_RULES}'
            })
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

    avg_loss = running_loss / len(train_loader)
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, logger=None):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data_tuple in test_loader:
            data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
            if target.ndim == 2 and target.shape[1] == 1: target = target.squeeze(1)

            with autocast(device_type=cfg.DEVICE.type, enabled=(cfg.DEVICE.type == 'cuda')):
                if cfg.MODEL_TYPE == 'DFM_FNCN':
                    output = model(data, labels=target, training_phase=False)
                else:
                    output = model(data)
                loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss History'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title('Accuracy History'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def visualize_and_save_rules(model, save_path, class_names):
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    if num_rules == 0: return

    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
    consequents_prob = scipy.special.softmax(consequents, axis=1)

    plt.figure(figsize=(12, max(8, num_rules * 0.3)))
    sns.heatmap(consequents_prob, annot=False, cmap='Reds',
                xticklabels=class_names, yticklabels=[f"Rule {i}" for i in range(num_rules)])
    plt.title('Fuzzy Rules Consequents (Probability)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fuzzy_rules_consequents.png'))
    plt.close()

def visualize_attention_weights(model, save_path):
    """[创新点 1] 可视化 Attention 权重 (Alpha)"""
    if not cfg.USE_ATTENTION or cfg.MODEL_TYPE != 'DFM_FNCN': return
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    if num_rules == 0 or classifier.alpha is None: return

    print("正在生成 Attention 权重热图...")
    alpha = classifier.alpha[:num_rules].detach().cpu().numpy()
    att_weights = scipy.special.softmax(alpha, axis=1)

    plt.figure(figsize=(20, max(8, num_rules * 0.3)))
    sns.heatmap(att_weights, annot=False, cmap='viridis',
                xticklabels=10, yticklabels=[f"Rule {i}" for i in range(num_rules)])
    plt.title('Fuzzy Rules Channel Attention Weights (Alpha)')
    plt.xlabel('Feature Channel Index')
    plt.ylabel('Rule Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fuzzy_rules_attention.png'))
    plt.close()

def visualize_rule_merging_history(merging_history, save_path):
    """[创新点 6] 可视化规则融合历史"""
    if not merging_history:
        return

    epochs = [entry['epoch'] for entry in merging_history]
    merged_counts = [entry['merged_count'] for entry in merging_history]
    rule_counts = [entry['rule_count'] for entry in merging_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 融合数量图
    axes[0].plot(epochs, merged_counts, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Merged Rules Count')
    axes[0].set_title('Rule Merging History')
    axes[0].grid(True, alpha=0.3)

    # 规则数量变化图
    axes[1].plot(epochs, rule_counts, 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Active Rules Count')
    axes[1].set_title('Active Rules Count History')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rule_merging_history.png'))
    plt.close()
    print(f"规则融合历史图已保存至: {os.path.join(save_path, 'rule_merging_history.png')}")

import shutil

def save_hyperparameters(save_path):
    """[新增] 保存超参数到 txt 文件和 config.py"""
    hp_file = os.path.join(save_path, 'hyperparameters.txt')
    with open(hp_file, 'w', encoding='utf-8') as f:
        f.write("==========================================\n")
        f.write(f"Training Run: {os.path.basename(save_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==========================================\n\n")

        f.write("[Experiment Control]\n")
        f.write(f"DATASET_NAME: {cfg.DATASET_NAME}\n")
        f.write(f"MODEL_TYPE: {cfg.MODEL_TYPE}\n")
        f.write(f"EXTRACTOR_TYPE: {cfg.EXTRACTOR_TYPE}\n")
        if cfg.DATASET_NAME == 'GTSRB':
             f.write(f"GTSRB_SUBSET_INDICES: {cfg.GTSRB_SUBSET_INDICES}\n")
        f.write("\n")

        f.write("[Training Config]\n")
        f.write(f"BATCH_SIZE: {cfg.BATCH_SIZE}\n")
        f.write(f"EPOCHS: {cfg.EPOCHS}\n")
        f.write(f"LR: {cfg.LR}\n")
        f.write(f"SEED: {cfg.SEED}\n")
        f.write("\n")

        f.write("[Model Config]\n")
        f.write(f"MAX_RULES: {cfg.MAX_RULES}\n")
        f.write(f"PHI_TH: exp({np.log(cfg.PHI_TH):.4f})\n")
        f.write(f"INIT_SIGMA: {cfg.INIT_SIGMA}\n")
        f.write(f"USE_ATTENTION: {cfg.USE_ATTENTION}\n")
        f.write(f"N_CHANNELS_OUT: {cfg.N_CHANNELS_OUT}\n")
        f.write("\n")

        f.write("[Clustering Init]\n")
        f.write(f"USE_CLUSTERING_INIT: {cfg.USE_CLUSTERING_INIT}\n")
        f.write(f"N_CLUSTERS: {cfg.N_CLUSTERS}\n")
        f.write(f"CLUSTERING_SAMPLE_LIMIT: {cfg.CLUSTERING_SAMPLE_LIMIT}\n")

        f.write("\n[Pruning Config]\n")
        f.write(f"USE_PRUNING: {cfg.USE_PRUNING}\n")
        f.write(f"PRUNING_METHOD: {cfg.PRUNING_METHOD}\n")
        f.write(f"PRUNING_THRESHOLD: {cfg.PRUNING_THRESHOLD}\n")

        f.write("\n[Rule Merging Config]\n")
        f.write(f"USE_RULE_MERGING: {cfg.USE_RULE_MERGING}\n")
        f.write(f"MERGING_METHOD: {cfg.MERGING_METHOD}\n")
        f.write(f"MERGING_THRESHOLD: {cfg.MERGING_THRESHOLD}\n")
        f.write(f"MERGING_STRATEGY: {cfg.MERGING_STRATEGY}\n")
        f.write(f"MERGING_TIMING: {cfg.MERGING_TIMING}\n")

    print(f"超参数已保存至: {hp_file}")

    # [新增] 直接复制当前 config.py 到保存路径
    config_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    config_dst = os.path.join(save_path, 'config.py')
    shutil.copy2(config_src, config_dst)
    print(f"config.py 已保存至: {config_dst}")

def run_training():
    set_seed(cfg.SEED)
    cfg.print_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.DATASET_NAME}_{cfg.MODEL_TYPE}_{cfg.EXTRACTOR_TYPE}_{timestamp}"
    SAVE_PATH = os.path.join('./checkpoints', run_name)
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    print(f"所有结果将保存到: {SAVE_PATH}")

    # [新增] 设置日志记录器
    logger = setup_logger(SAVE_PATH)
    logger.info("=" * 60)
    logger.info(f"开始训练: {run_name}")
    logger.info(f"保存路径: {SAVE_PATH}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 记录配置信息
    logger.info(f"数据集: {cfg.DATASET_NAME}")
    logger.info(f"模型类型: {cfg.MODEL_TYPE}")
    logger.info(f"特征提取器: {cfg.EXTRACTOR_TYPE}")
    logger.info(f"批次大小: {cfg.BATCH_SIZE}")
    logger.info(f"训练轮数: {cfg.EPOCHS}")
    logger.info(f"学习率: {cfg.LR}")
    logger.info(f"随机种子: {cfg.SEED}")
    if cfg.DATASET_NAME == 'GTSRB' and cfg.GTSRB_SUBSET_INDICES is not None:
        logger.info(f"GTSRB子集: {cfg.GTSRB_SUBSET_INDICES}")
    logger.info(f"注意力机制: {cfg.USE_ATTENTION}")
    logger.info(f"聚类初始化: {cfg.USE_CLUSTERING_INIT}")
    logger.info(f"规则修剪: {cfg.USE_PRUNING}")
    logger.info(f"规则融合: {cfg.USE_RULE_MERGING}")

    # [新增] 保存超参数
    save_hyperparameters(SAVE_PATH)

    # [修改] 获取数据加载器和类别权重
    train_loader, test_loader, class_weights = get_data_loaders()
    logger.info(f"训练集大小: {len(train_loader.dataset)}") # type: ignore
    logger.info(f"测试集大小: {len(test_loader.dataset)}") # type: ignore
    logger.info(f"类别权重: {class_weights.tolist()}")

    if cfg.MODEL_TYPE == 'DFM_FNCN': model = FullModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'TRADITIONAL_CNN': model = TraditionalCNNModel().to(cfg.DEVICE)
    else: raise ValueError(f"未知的 MODEL_TYPE: {cfg.MODEL_TYPE}")

    # [创新点 3] 执行聚类初始化
    if cfg.MODEL_TYPE == 'DFM_FNCN' and cfg.USE_CLUSTERING_INIT:
        perform_clustering_initialization(model, train_loader, logger)
        logger.info(f"提示: 已启用聚类初始化。动态规则生成保持开启 (PHI_TH = {cfg.PHI_TH:.2e})。")

    # [修改] 使用加权 CrossEntropyLoss 解决类别不平衡
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))
    logger.info("已应用类别权重 (Class Weights) 以解决样本不平衡问题。")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scaler = GradScaler(device='cuda', enabled=(cfg.DEVICE.type == 'cuda')) # type: ignore

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    best_model_save_path = os.path.join(SAVE_PATH, 'best_model.pth')

    # [创新点 6] 记录规则融合历史
    merging_history = []

    logger.info(f"\n开始训练 {cfg.MODEL_TYPE}...")
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start_time = time.time()

    for epoch in range(cfg.EPOCHS):
        epoch_start_time = datetime.now()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler, logger)
        test_loss, test_acc = evaluate(model, test_loader, criterion, logger)

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss); history['test_acc'].append(test_acc)

        epoch_log = f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
        print(epoch_log)
        logger.info(epoch_log)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_rules': cfg.MAX_RULES,
                'config_params': {
                    'MODEL_TYPE': cfg.MODEL_TYPE,
                    'EXTRACTOR_TYPE': cfg.EXTRACTOR_TYPE,
                    'DATASET_NAME': cfg.DATASET_NAME,
                    'USE_ATTENTION': cfg.USE_ATTENTION,
                    'USE_CLUSTERING_INIT': cfg.USE_CLUSTERING_INIT,
                    'USE_PRUNING': cfg.USE_PRUNING,
                    'USE_RULE_MERGING': cfg.USE_RULE_MERGING,
                    'GTSRB_SUBSET_INDICES': cfg.GTSRB_SUBSET_INDICES,
                    'INIT_SIGMA': cfg.INIT_SIGMA,
                    'PHI_TH': cfg.PHI_TH
                }
            }, best_model_save_path)
            best_msg = f"    *** 新最佳权重 (Acc: {best_acc:.2f}%) ***"
            print(best_msg)
            logger.info(best_msg)

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            active_rules = model.classifier.num_active_rules.item() # type: ignore
            rules_msg = f"    Active Rules: {active_rules}/{cfg.MAX_RULES}"
            print(rules_msg)
            logger.info(rules_msg)

            # [创新点 6] 执行规则融合（如果配置为每个epoch后融合）
            if cfg.USE_RULE_MERGING and cfg.MERGING_TIMING == 'EVERY_EPOCH':
                merged_count, merge_pairs = perform_rule_merging(model, test_loader, epoch, logger)
                if merged_count > 0:
                    merging_history.append({
                        'epoch': epoch + 1,
                        'merged_count': merged_count,
                        'merge_pairs': merge_pairs,
                        'rule_count': model.classifier.num_active_rules.item() # type: ignore
                    })
                    merge_msg = f"    [Merging] 融合了 {merged_count} 对规则，当前规则数: {model.classifier.num_active_rules.item()}" # type: ignore
                    logger.info(merge_msg)

        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        logger.info(f"Epoch {epoch+1} 完成，耗时: {epoch_duration:.2f}秒")

    # 记录最终训练结果
    total_time = time.time() - total_start_time
    final_msg = f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%"
    print(final_msg)
    logger.info(final_msg)
    logger.info(f"训练历史 - 最终训练准确率: {history['train_acc'][-1]:.2f}%")
    logger.info(f"训练历史 - 最终测试准确率: {history['test_acc'][-1]:.2f}%")
    logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

    # [创新点 2] 训练结束后执行规则修剪
    if cfg.MODEL_TYPE == 'DFM_FNCN' and cfg.USE_PRUNING:
        logger.info("\n>>> 训练结束，开始执行规则修剪...")
        checkpoint = torch.load(best_model_save_path,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 保存修剪前的完整模型（加后缀complete）
        complete_model_save_path = os.path.join(SAVE_PATH, 'best_model_complete.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint['config_params']
        }, complete_model_save_path)
        logger.info(f"修剪前的完整模型已保存至: {complete_model_save_path}")

        perform_rule_pruning(model, test_loader, logger)

        pruned_model_save_path = os.path.join(SAVE_PATH, 'best_model_pruned.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint['config_params']
        }, pruned_model_save_path)
        logger.info(f"修剪后的模型已保存至: {pruned_model_save_path}")

        pruned_loss, pruned_acc = evaluate(model, test_loader, criterion, logger)
        prune_result = f"修剪后性能: Loss {pruned_loss:.4f}, Acc {pruned_acc:.2f}%"
        print(prune_result)
        logger.info(prune_result)

        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint['config_params']
        }, best_model_save_path)
        logger.info("已更新最佳模型文件为修剪后版本。")

    # [创新点 6] 训练结束后执行规则融合（如果配置为只在训练结束时融合）
    if cfg.MODEL_TYPE == 'DFM_FNCN' and cfg.USE_RULE_MERGING and cfg.MERGING_TIMING == 'FINAL_ONLY':
        logger.info("\n>>> 训练结束，开始执行规则融合...")
        merged_count, merge_pairs = perform_rule_merging(model, test_loader, logger=logger)
        if merged_count > 0:
            merging_history.append({
                'epoch': cfg.EPOCHS,
                'merged_count': merged_count,
                'merge_pairs': merge_pairs,
                'rule_count': model.classifier.num_active_rules.item() # type: ignore
            })

            # 保存融合后的模型
            merged_model_save_path = os.path.join(SAVE_PATH, 'best_model_merged.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_rules': cfg.MAX_RULES,
                'config_params': checkpoint['config_params'] # type: ignore
            }, merged_model_save_path)
            logger.info(f"融合后的模型已保存至: {merged_model_save_path}")

            merged_loss, merged_acc = evaluate(model, test_loader, criterion, logger)
            merge_result = f"融合后性能: Loss {merged_loss:.4f}, Acc {merged_acc:.2f}%"
            print(merge_result)
            logger.info(merge_result)

    # 保存训练历史图表
    logger.info("\n>>> 正在生成训练历史图表...")
    plot_history(history, SAVE_PATH)
    logger.info(f"训练历史图表已保存至: {os.path.join(SAVE_PATH, 'training_history.png')}")

    # 如果是 DFM_FNCN 模型，生成规则可视化
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        logger.info("\n>>> 正在生成规则可视化...")
        visualize_and_save_rules(model, SAVE_PATH, cfg.CLASS_NAMES)
        logger.info(f"规则后件热图已保存至: {os.path.join(SAVE_PATH, 'fuzzy_rules_consequents.png')}")

        visualize_attention_weights(model, SAVE_PATH)
        logger.info(f"注意力权重热图已保存至: {os.path.join(SAVE_PATH, 'fuzzy_rules_attention.png')}")

        # [创新点 6] 可视化规则融合历史
        if merging_history:
            visualize_rule_merging_history(merging_history, SAVE_PATH)
            logger.info(f"规则融合历史图已保存至: {os.path.join(SAVE_PATH, 'rule_merging_history.png')}")

        # 记录最终规则数量
        final_rules = model.classifier.num_active_rules.item() # type: ignore
        logger.info(f"最终激活规则数量: {final_rules}/{cfg.MAX_RULES}")
        logger.info(f"规则压缩率: {100 * (1 - final_rules / cfg.MAX_RULES):.1f}%")

    # 记录训练总结
    logger.info("\n" + "=" * 60)
    logger.info("训练总结:")
    logger.info(f"数据集: {cfg.DATASET_NAME}")
    logger.info(f"模型类型: {cfg.MODEL_TYPE}")
    logger.info(f"特征提取器: {cfg.EXTRACTOR_TYPE}")
    logger.info(f"最佳测试准确率: {best_acc:.2f}%")
    logger.info(f"最终训练准确率: {history['train_acc'][-1]:.2f}%")
    logger.info(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        logger.info(f"最终规则数量: {model.classifier.num_active_rules.item()}") # type: ignore
        logger.info(f"规则融合次数: {len(merging_history)}")
    logger.info("=" * 60)

    # 关闭日志处理器
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    total_time = time.time() - total_start_time
    print(f"\n训练完成! 所有结果已保存到: {SAVE_PATH}")
    print(f"训练日志已保存到: {os.path.join(SAVE_PATH, 'training_log.txt')}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

    return SAVE_PATH

if __name__ == '__main__':
    run_training()