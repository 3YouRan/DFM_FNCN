import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from datetime import datetime
import medmnist
from medmnist import BloodMNIST
from torch.cuda.amp import autocast, GradScaler
import scipy.special

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
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    print(f"成功加载 {cfg.DATASET_NAME} 数据集。")
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, data_tuple in enumerate(train_loader):
        data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
        if target.ndim == 2 and target.shape[1] == 1: target = target.squeeze(1)

        optimizer.zero_grad()
        with autocast(enabled=(cfg.DEVICE.type == 'cuda')):
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

            with autocast(enabled=(cfg.DEVICE.type == 'cuda')):
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

def main():
    set_seed(cfg.SEED)
    cfg.print_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.DATASET_NAME}_{cfg.MODEL_TYPE}_{cfg.EXTRACTOR_TYPE}_{timestamp}"
    SAVE_PATH = os.path.join('./checkpoints', run_name)
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    print(f"所有结果将保存到: {SAVE_PATH}")

    train_loader, test_loader = get_data_loaders()
    if cfg.MODEL_TYPE == 'DFM_FNCN': model = FullModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'TRADITIONAL_CNN': model = TraditionalCNNModel().to(cfg.DEVICE)
    else: raise ValueError(f"未知的 MODEL_TYPE: {cfg.MODEL_TYPE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scaler = GradScaler(enabled=(cfg.DEVICE.type == 'cuda'))

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
                    'USE_ATTENTION': cfg.USE_ATTENTION # 保存配置
                }
            }, best_model_save_path)
            print(f"    *** 新最佳权重 (Acc: {best_acc:.2f}%) ***")

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            print(f"    Active Rules: {model.classifier.num_active_rules.item()}/{cfg.MAX_RULES}")

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")
    plot_history(history, SAVE_PATH)
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        visualize_and_save_rules(model, SAVE_PATH, cfg.CLASS_NAMES)
        visualize_attention_weights(model, SAVE_PATH)

if __name__ == '__main__':
    main()