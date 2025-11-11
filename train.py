import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

# 导入我们的自定义模块
import config as cfg
from models import FullModel


# ==========================================
# 工具函数
# ==========================================
def set_seed(seed):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_loaders():
    """准备 Fashion-MNIST 数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])

    train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    # 测试集 batch_size 可以大一些，加快推理速度
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

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

        # [重要] 传入 labels 和 training_phase=True 以启用动态规则生成
        output = model(data, labels=target, training_phase=True)

        loss = criterion(output, target)
        loss.backward()

        # [可选] 梯度裁剪，防止新加入规则初期梯度过大导致不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

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
            # 测试时关闭规则生成
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

    # 1. 数据准备
    print("正在加载数据...")
    train_loader, test_loader = get_data_loaders()

    # 2. 模型初始化
    model = FullModel().to(cfg.DEVICE)

    # 3. 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器会维护所有参数（包括尚未激活的规则）的状态。
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # 4. 训练循环
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
            # 可选：保存最佳模型
            # torch.save(model.state_dict(), 'best_dfm_fncn.pth')

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()