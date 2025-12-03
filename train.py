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
import medmnist
from medmnist import BloodMNIST
# [修改] 更新 AMP 导入
from torch.amp import autocast, GradScaler
import scipy.special
# [创新点 3] 导入 KMeans
from sklearn.cluster import KMeans

import config as cfg
from models import FullModel, TraditionalCNNModel

CLASS_NAMES = cfg.CLASS_NAMES

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
    if cfg.IN_CHANNELS == 1:
        norm_mean, norm_std = (0.5,), (0.5,)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    data_transform = transforms.Compose([
        transforms.Resize(cfg.TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    if cfg.DATASET_NAME == 'FASHION_MNIST':
        train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'SVHN':
        train_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform)
        test_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        train_dataset = BloodMNIST(split='train', transform=data_transform, download=True, root=cfg.DATA_ROOT)
        test_dataset = BloodMNIST(split='test', transform=data_transform, download=True, root=cfg.DATA_ROOT)
    elif cfg.DATASET_NAME == 'GTSRB':
        # [修改] GTSRB 子集处理逻辑
        target_transform = None
        if cfg.GTSRB_SUBSET_INDICES is not None:
            # 1. 创建标签映射: 原始ID -> 0..N-1
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(cfg.GTSRB_SUBSET_INDICES)}
            # 如果标签不在子集中，返回 -1 (后续会被过滤掉)
            target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))

        train_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='train', download=True,
                                       transform=data_transform, target_transform=target_transform)
        test_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='test', download=True,
                                      transform=data_transform, target_transform=target_transform)

        if cfg.GTSRB_SUBSET_INDICES is not None:
            # 2. 过滤数据集，只保留子集中的样本
            # 使用 _samples (list of (path, class_id)) 快速筛选，避免加载图片
            subset_set = set(cfg.GTSRB_SUBSET_INDICES)
            train_indices = [i for i, (_, label) in enumerate(train_dataset._samples) if label in subset_set]
            test_indices = [i for i, (_, label) in enumerate(test_dataset._samples) if label in subset_set]

            train_dataset = Subset(train_dataset, train_indices)
            test_dataset = Subset(test_dataset, test_indices)
            print(f"GTSRB Subset: Train {len(train_dataset)}, Test {len(test_dataset)}")
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    print(f"成功加载 {cfg.DATASET_NAME} 数据集。")
    return train_loader, test_loader

def perform_clustering_initialization(model, train_loader):
    """[创新点 3] 使用 K-Means 聚类初始化规则中心"""
    print(f"\n>>> 正在执行聚类初始化 (K-Means, K={cfg.N_CLUSTERS})...")
    model.eval()
    features_list = []
    labels_list = []

    # 1. 收集特征
    print("正在收集样本特征...")
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
    all_labels = np.concatenate(labels_list, axis=0)

    print(f"收集了 {all_features.shape[0]} 个样本用于聚类。")

    # 2. 执行聚类
    print("正在运行 K-Means...")
    kmeans = KMeans(n_clusters=cfg.N_CLUSTERS, n_init=10, random_state=cfg.SEED)
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
            # 多数投票
            counts = np.bincount(k_labels, minlength=cfg.N_CLASSES)
            majority_class = np.argmax(counts)
            cluster_majority_classes.append(majority_class)
        else:
            cluster_majority_classes.append(0)  # Fallback

    # 4. 将中心和类别传回模型
    # Reshape centers back to (K, C, P_Dim)
    # Feature_Dim = C * P_Dim.
    # In model: x_flat = x.view(b, self.n_channels, -1) -> (B, C, P)
    # Here we flattened as (B, C*P).
    # So reshaping (K, C*P) -> (K, C, P) works if C is the first dimension after batch.

    cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32).to(cfg.DEVICE)
    cluster_centers_reshaped = cluster_centers_tensor.view(cfg.N_CLUSTERS, cfg.N_CHANNELS_OUT, cfg.P_DIM)

    model.classifier.init_rules_from_cluster_centers(cluster_centers_reshaped, cluster_majority_classes)
    print("聚类初始化完成。\n")

def perform_rule_pruning(model, test_loader):
    """[创新点 2] 执行规则修剪"""
    print(f"\n>>> 正在执行规则修剪 (Method: {cfg.PRUNING_METHOD}, Th: {cfg.PRUNING_THRESHOLD})...")
    model.eval()
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    if num_rules == 0:
        print("没有激活的规则，跳过修剪。")
        return

    keep_indices = []

    if cfg.PRUNING_METHOD == 'CONSEQUENT':
        # 方法 A: 基于后件置信度
        # 检查每条规则对类别的最大预测概率
        consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
        consequents_prob = scipy.special.softmax(consequents, axis=1)
        max_probs = np.max(consequents_prob, axis=1)

        for i in range(num_rules):
            if max_probs[i] >= cfg.PRUNING_THRESHOLD:
                keep_indices.append(i)
            else:
                print(f"    [Prune] Rule {i} dropped (Max Prob: {max_probs[i]:.4f} < {cfg.PRUNING_THRESHOLD})")

    elif cfg.PRUNING_METHOD == 'ACTIVATION':
        # 方法 B: 基于激活强度
        # 需要在测试集上运行一遍，统计每条规则的平均激活度
        print("正在计算测试集上的规则激活度...")
        total_activations = torch.zeros(num_rules, device=cfg.DEVICE)
        total_samples = 0

        with torch.no_grad():
            for data_tuple in test_loader:
                data = data_tuple[0].to(cfg.DEVICE)
                features = model.extractor(data)
                # 获取归一化的 phi (B, Rules)
                phi = classifier.get_rule_activations(features)
                if phi is not None:
                    total_activations += torch.sum(phi, dim=0)
                    total_samples += data.size(0)

        avg_activations = (total_activations / total_samples).cpu().numpy()

        for i in range(num_rules):
            if avg_activations[i] >= cfg.PRUNING_THRESHOLD:
                keep_indices.append(i)
            else:
                print(f"    [Prune] Rule {i} dropped (Avg Act: {avg_activations[i]:.5f} < {cfg.PRUNING_THRESHOLD})")

    else:
        print(f"未知的修剪方法: {cfg.PRUNING_METHOD}")
        return

    # 执行修剪
    if len(keep_indices) < num_rules:
        classifier.prune_rules(keep_indices)
    else:
        print("没有规则被修剪。")
    print("规则修剪完成。\n")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, data_tuple in enumerate(train_loader):
        data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
        if target.ndim == 2 and target.shape[1] == 1: target = target.squeeze(1)

        optimizer.zero_grad()
        # [修改] 使用 torch.amp.autocast 替代 deprecated API
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

        if (batch_idx + 1) % 200 == 0:
            log_msg = f"[Epoch {epoch + 1}] Step {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%"
            if cfg.MODEL_TYPE == 'DFM_FNCN':
                log_msg += f" | Active Rules: {model.classifier.num_active_rules.item()}"
            print(log_msg)

    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data_tuple in test_loader:
            data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
            if target.ndim == 2 and target.shape[1] == 1: target = target.squeeze(1)

            # [修改] 使用 torch.amp.autocast
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
    return test_loss / len(test_loader), 100. * correct / total

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

def save_hyperparameters(save_path):
    """[新增] 保存超参数到 txt 文件"""
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
        f.write(f"PHI_TH: {cfg.PHI_TH}\n")
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

    print(f"超参数已保存至: {hp_file}")

def run_training():
    set_seed(cfg.SEED)
    cfg.print_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.DATASET_NAME}_{cfg.MODEL_TYPE}_{cfg.EXTRACTOR_TYPE}_{timestamp}"
    SAVE_PATH = os.path.join('./checkpoints', run_name)
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    print(f"所有结果将保存到: {SAVE_PATH}")

    # [新增] 保存超参数
    save_hyperparameters(SAVE_PATH)

    train_loader, test_loader = get_data_loaders()
    if cfg.MODEL_TYPE == 'DFM_FNCN': model = FullModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'TRADITIONAL_CNN': model = TraditionalCNNModel().to(cfg.DEVICE)
    else: raise ValueError(f"未知的 MODEL_TYPE: {cfg.MODEL_TYPE}")

    # [创新点 3] 执行聚类初始化
    if cfg.MODEL_TYPE == 'DFM_FNCN' and cfg.USE_CLUSTERING_INIT:
        perform_clustering_initialization(model, train_loader)
        # [修改] 允许继续动态生成规则 (不再强制 phi_th = 0.0)
        print(f"提示: 已启用聚类初始化。动态规则生成保持开启 (PHI_TH = {cfg.PHI_TH:.2e})。")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    # [修改] 使用 torch.amp.GradScaler
    scaler = GradScaler(device='cuda', enabled=(cfg.DEVICE.type == 'cuda'))

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    best_model_save_path = os.path.join(SAVE_PATH, 'best_model.pth')

    print(f"\n开始训练 {cfg.MODEL_TYPE}...")
    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss); history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_rules': cfg.MAX_RULES,
                'config_params': {
                    'MODEL_TYPE': cfg.MODEL_TYPE,
                    'EXTRACTOR_TYPE': cfg.EXTRACTOR_TYPE,
                    'DATASET_NAME': cfg.DATASET_NAME,
                    'USE_ATTENTION': cfg.USE_ATTENTION
                }
            }, best_model_save_path)
            print(f"    *** 新最佳权重 (Acc: {best_acc:.2f}%) ***")

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            print(f"    Active Rules: {model.classifier.num_active_rules.item()}/{cfg.MAX_RULES}")

    # [创新点 2] 训练结束后执行规则修剪
    if cfg.MODEL_TYPE == 'DFM_FNCN' and cfg.USE_PRUNING:
        print("\n>>> 训练结束，开始执行规则修剪...")
        # 加载最佳模型进行修剪
        checkpoint = torch.load(best_model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        perform_rule_pruning(model, test_loader)

        # 保存修剪后的模型
        pruned_model_save_path = os.path.join(SAVE_PATH, 'best_model_pruned.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint['config_params']
        }, pruned_model_save_path)
        print(f"修剪后的模型已保存至: {pruned_model_save_path}")

        # 重新评估修剪后的模型
        pruned_loss, pruned_acc = evaluate(model, test_loader, criterion)
        print(f"修剪后性能: Loss {pruned_loss:.4f}, Acc {pruned_acc:.2f}%")

        # 更新 best_model.pth 为修剪后的版本 (可选，这里选择覆盖以供后续步骤使用)
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint['config_params']
        }, best_model_save_path)
        print("已更新最佳模型文件为修剪后版本。")

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")
    plot_history(history, SAVE_PATH)
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        visualize_and_save_rules(model, SAVE_PATH, cfg.CLASS_NAMES)
        visualize_attention_weights(model, SAVE_PATH)

    return SAVE_PATH

if __name__ == '__main__':
    run_training()