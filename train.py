import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

import config as cfg
from models import FullModel


# ==========================================
# 工具函数 (保持不变)
# ==========================================
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=transform)
    # 根据您的硬件情况，可能需要调整 num_workers。如果报错，可以设为 0。
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ==========================================
# 训练与评估流程
# ==========================================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)

        optimizer.zero_grad()

        # 前向传播 (可能会检测到需要新规则，并将其存入 pending_rule_data)
        output = model(data, labels=target, training_phase=True)

        loss = criterion(output, target)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # [关键修改] 在优化器更新完参数后，安全地提交新规则
        # 此时修改参数不会影响已经完成的 backward 过程
        model.classifier.commit_pending_rule()

        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 200 == 0:
            current_rules = model.classifier.num_active_rules
            print(f"[Epoch {epoch + 1}] Step {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}% | "
                  f"Active Rules: {current_rules}")

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            output = model(data, labels=target, training_phase=False)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return test_loss / len(test_loader), 100. * correct / total


# ==========================================
# 主函数
# ==========================================
def main():
    set_seed(cfg.SEED)
    cfg.print_config()

    print("正在加载数据...")
    train_loader, test_loader = get_data_loaders()

    model = FullModel().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    print("\n开始训练 DFM-FNCN...")
    best_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        current_rules = model.classifier.num_active_rules
        print(f"\n==> Epoch {epoch + 1} 完成.")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%")
        print(f"    当前规则总数: {current_rules}/{cfg.MAX_RULES}\n")

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()