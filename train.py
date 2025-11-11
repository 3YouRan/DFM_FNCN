import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题、
import seaborn as sns  # 用于绘制漂亮的热图
import torch.nn.functional as F

# 导入我们的自定义模块
import config as cfg
from models import FullModel


# ==========================================
# 辅助函数 (保持不变)
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

        output = model(data, labels=target, training_phase=True)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        model.classifier.commit_pending_rule()

        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 200 == 0:
            # [关键修复] 使用 .item() 读取缓冲区的值
            current_rules = model.classifier.num_active_rules.item()
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
# 结果保存与可视化
# ==========================================
def plot_history(history, save_path):
    print("正在绘制训练历史图表...")
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['train_loss'], label='训练损失 (Train Loss)', color='blue')
    ax1.plot(history['test_loss'], label='测试损失 (Test Loss)', color='orange')
    ax1.set_title("损失函数曲线 (Loss History)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失 (Loss)")
    ax1.legend()

    ax2.plot(history['train_acc'], label='训练准确率 (Train Acc)', color='blue')
    ax2.plot(history['test_acc'], label='测试准确率 (Test Acc)', color='orange')
    ax2.set_title("准确率曲线 (Accuracy History)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率 (%)")
    ax2.legend()

    plt.suptitle("DFM-FNCN 训练统计", fontsize=16)
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def visualize_and_save_rules(model, save_path):
    print("正在保存和可视化模糊规则...")
    model.eval()
    classifier = model.classifier
    # [关键修复] 使用 .item() 读取缓冲区的值
    active_rules = classifier.num_active_rules.item()

    if active_rules == 0:
        print("警告: 模型中没有激活的规则。")
        return

    # 1. 提取所有激活的规则参数
    centers = classifier.centers.detach().cpu()[:active_rules]
    widths = F.softplus(classifier.widths_param).detach().cpu()[:active_rules]
    consequents = classifier.consequents.detach().cpu()[:active_rules]

    # 2. 保存原始规则数据
    rules_data = {
        'centers': centers,
        'widths': widths,
        'consequents': consequents,
        'num_active_rules': active_rules  # 保存 Python 整数值
    }
    torch.save(rules_data, os.path.join(save_path, 'fuzzy_rules_data.pth'))

    # 3. 可视化规则后件 (Consequents)
    consequents_softmax = F.softmax(consequents, dim=1)

    h = max(10, active_rules // 5)
    plt.figure(figsize=(12, h))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    sns.heatmap(consequents_softmax, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=[f"R{i}" for i in range(active_rules)])

    plt.title(f"模糊规则后件可视化 (共 {active_rules} 条规则)\n每条规则对各类的预测倾向", fontsize=16)
    plt.xlabel("类别 (Class)", fontsize=12)
    plt.ylabel("规则 ID (Rule #)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fuzzy_rules_consequents.png'))
    plt.close()

    print(f"总共生成 {active_rules} 条模糊规则。")
    print(f"规则数据已保存到 {save_path}/fuzzy_rules_data.pth")
    print(f"规则后件可视化已保存到 {save_path}/fuzzy_rules_consequents.png")

    # 4. 打印规则解释
    print("\n--- 规则可解释性分析 (后件) ---")
    try:
        strongest_rule_for_class = consequents_softmax.argmax(dim=0)
        for i, rule_idx in enumerate(strongest_rule_for_class):
            print(f" -> 类别 '{class_names[i]}' 的最强规则是: 规则 #{rule_idx.item()}")
    except Exception as e:
        print(f"打印规则分析时出错: {e}")


# ==========================================
# 主函数
# ==========================================
def main():
    set_seed(cfg.SEED)
    cfg.print_config()

    SAVE_PATH = './checkpoints'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    print("正在加载数据...")
    train_loader, test_loader = get_data_loaders()

    print("正在初始化模型...")
    model = FullModel().to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0

    print("\n开始训练 DFM-FNCN...")
    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"\n==> Epoch {epoch + 1} 完成.")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_file = os.path.join(SAVE_PATH, 'best_model.pth')
            torch.save(model.state_dict(), save_file)
            print(f"    *** 新的最佳权重已保存至 {save_file} (Acc: {best_acc:.2f}%) ***")

        # [关键修复] 使用 .item() 读取缓冲区的值
        current_rules = model.classifier.num_active_rules.item()
        print(f"    当前规则总数: {current_rules}/{cfg.MAX_RULES}\n")

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")

    plot_history(history, SAVE_PATH)
    visualize_and_save_rules(model, SAVE_PATH)


if __name__ == '__main__':
    main()