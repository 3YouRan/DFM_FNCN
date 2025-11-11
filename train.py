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
# [新] 导入自动混合精度 (AMP)
from torch.cuda.amp import autocast, GradScaler

# 导入我们的自定义模块
import config as cfg
from models import FullModel, TraditionalCNNModel

# CLASS_NAMES 现在从 config 动态加载
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
    """根据 config.py 动态加载数据集"""

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
        train_dataset = datasets.FashionMNIST(
            root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform
        )
        test_dataset = datasets.FashionMNIST(
            root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'SVHN':
        train_dataset = datasets.SVHN(
            root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform
        )
        test_dataset = datasets.SVHN(
            root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        train_dataset = BloodMNIST(
            split='train', transform=data_transform, download=True, root=cfg.DATA_ROOT
        )
        test_dataset = BloodMNIST(
            split='test', transform=data_transform, download=True, root=cfg.DATA_ROOT
        )
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    print(f"成功加载 {cfg.DATASET_NAME} 数据集。")
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler):
    """[新] 增加了 scaler 参数"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, data_tuple in enumerate(train_loader):
        data, target = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)

        if target.ndim == 2 and target.shape[1] == 1:
            target = target.squeeze(1)

        optimizer.zero_grad()

        # [新] 使用 autocast 包装器
        # 它会自动将 CUDA 操作转换为 float16
        with autocast(enabled=(cfg.DEVICE.type == 'cuda')):
            if cfg.MODEL_TYPE == 'DFM_FNCN':
                output = model(data, labels=target, training_phase=True)
            else:  # 'TRADITIONAL_CNN'
                output = model(data)
            loss = criterion(output, target)

        # [新] scaler.scale(loss) 会缩放损失值
        scaler.scale(loss).backward()

        # [新] scaler.step() 会自动 unscale 梯度
        scaler.step(optimizer)

        # [新] 更新 scaler
        scaler.update()

        # 梯度裁剪 (在 scaler 之后执行是安全的)
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
            if target.ndim == 2 and target.shape[1] == 1:
                target = target.squeeze(1)

            # [新] 在评估时也使用 autocast (无需 scaler) 以提高效率
            with autocast(enabled=(cfg.DEVICE.type == 'cuda')):
                if cfg.MODEL_TYPE == 'DFM_FNCN':
                    output = model(data, labels=target, training_phase=False)
                else:  # 'TRADITIONAL_CNN'
                    output = model(data)

                loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader), 100. * correct / total


def plot_history(history, save_path):
    # ... (此函数无变化)
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def visualize_and_save_rules(model, save_path, class_names):
    # ... (此函数无变化)
    plt.savefig(os.path.join(save_path, 'fuzzy_rules_consequents.png'))
    plt.close()
    print(f"规则后件可视化已保存到 {os.path.join(save_path, 'fuzzy_rules_consequents.png')}")


def main():
    set_seed(cfg.SEED)
    cfg.print_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.DATASET_NAME}_{cfg.MODEL_TYPE}_{cfg.EXTRACTOR_TYPE}_{timestamp}"
    SAVE_PATH = os.path.join('./checkpoints', run_name)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"所有结果将保存到: {SAVE_PATH}")

    print("正在加载数据...")
    train_loader, test_loader = get_data_loaders()

    print("正在初始化模型...")
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        model = FullModel().to(cfg.DEVICE)
    elif cfg.MODEL_TYPE == 'TRADITIONAL_CNN':
        model = TraditionalCNNModel().to(cfg.DEVICE)
    else:
        raise ValueError(f"未知的 MODEL_TYPE: {cfg.MODEL_TYPE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # [新] 初始化 GradScaler
    # 仅在 CUDA 可用时启用
    scaler = GradScaler(enabled=(cfg.DEVICE.type == 'cuda'))
    print(f"自动混合精度 (AMP) 启用: {scaler.is_enabled()}")

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    best_model_save_path = os.path.join(SAVE_PATH, 'best_model.pth')

    print(f"\n开始训练 {cfg.MODEL_TYPE} ({cfg.EXTRACTOR_TYPE}) on {cfg.DATASET_NAME}...")
    for epoch in range(cfg.EPOCHS):
        # [新] 传入 scaler
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss);
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss);
        history['test_acc'].append(test_acc)

        print(f"\n==> Epoch {epoch + 1} 完成.")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_rules': cfg.MAX_RULES,
                'config_params': {
                    'MODEL_TYPE': cfg.MODEL_TYPE,
                    'EXTRACTOR_TYPE': cfg.EXTRACTOR_TYPE,
                    'DATASET_NAME': cfg.DATASET_NAME
                }
            }, best_model_save_path)
            print(f"    *** 新的最佳权重已保存至 {best_model_save_path} (Acc: {best_acc:.2f}%) ***")

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            print(f"    当前规则总数: {model.classifier.num_active_rules.item()}/{cfg.MAX_RULES}\n")
        else:
            print("\n")

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")
    plot_history(history, SAVE_PATH)

    if cfg.MODEL_TYPE == 'DFM_FNCN':
        visualize_and_save_rules(model, SAVE_PATH, cfg.CLASS_NAMES)


if __name__ == '__main__':
    main()