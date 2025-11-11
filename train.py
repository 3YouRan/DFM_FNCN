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
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import seaborn as sns
import torch.nn.functional as F
from datetime import datetime  # [新] 导入 datetime

# 导入我们的自定义模块
import config as cfg
from models import FullModel, TraditionalCNNModel


# ==========================================
# 辅助函数 (保持不变)
# ==========================================
def set_seed(seed):
    # ... (保持不变) ...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_loaders():
    # ... (保持不变) ...
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
# 训练与评估流程 (保持不变)
# ==========================================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    # ... (保持不变) ...
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
        optimizer.zero_grad()

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            output = model(data, labels=target, training_phase=True)
        else:  # 'TRADITIONAL_CNN'
            output = model(data)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            model.classifier.commit_pending_rule()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 200 == 0:
            log_msg = (
                f"[Epoch {epoch + 1}] Step {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%"
            )
            if cfg.MODEL_TYPE == 'DFM_FNCN':
                current_rules = model.classifier.num_active_rules.item()
                log_msg += f" | Active Rules: {current_rules}"
            print(log_msg)

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    # ... (保持不变) ...
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)

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


# ==========================================
# 结果保存与可视化
# ==========================================
def plot_history(history, save_path):
    print("正在绘制训练历史图表...")
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['train_loss'], label='训练损失', color='blue')
    ax1.plot(history['test_loss'], label='测试损失', color='orange')
    ax1.set_title("损失函数曲线 (Loss History)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失 (Loss)")
    ax1.legend()

    ax2.plot(history['train_acc'], label='训练准确率', color='blue')
    ax2.plot(history['test_acc'], label='测试准确率', color='orange')
    ax2.set_title("准确率曲线 (Accuracy History)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率 (%)")
    ax2.legend()

    plt.suptitle(f"模型: {cfg.MODEL_TYPE} | 提取器: {cfg.EXTRACTOR_TYPE}", fontsize=16)
    # [修改] 保存到唯一的运行目录中
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()


def visualize_and_save_rules(model, save_path):
    # ... (保持不变) ...
    if cfg.MODEL_TYPE != 'DFM_FNCN' or not hasattr(model, 'classifier'):
        print("非 DFM-FNCN 模型，跳过规则可视化。")
        return

    print("正在保存和可视化模糊规则...")
    model.eval()
    classifier = model.classifier
    active_rules = classifier.num_active_rules.item()

    if active_rules == 0:
        print("警告: 模型中没有激活的规则。")
        return

    centers = classifier.centers.detach().cpu()[:active_rules]
    widths = F.softplus(classifier.widths_param).detach().cpu()[:active_rules]
    consequents = classifier.consequents.detach().cpu()[:active_rules]

    rules_data = {
        'centers': centers, 'widths': widths, 'consequents': consequents,
        'num_active_rules': active_rules
    }
    # [修改] 保存到唯一的运行目录中
    torch.save(rules_data, os.path.join(save_path, 'fuzzy_rules_data.pth'))

    consequents_softmax = F.softmax(consequents, dim=1)

    h = max(10, active_rules // 5)
    plt.figure(figsize=(12, h))
    class_names = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    sns.heatmap(consequents_softmax, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=[f"R{i}" for i in range(active_rules)])

    plt.title(f"模糊规则后件可视化 (共 {active_rules} 条规则)", fontsize=16)
    plt.xlabel("类别 (Class)", fontsize=12)
    plt.ylabel("规则 ID (Rule #)", fontsize=12)
    plt.tight_layout()
    # [修改] 保存到唯一的运行目录中
    plt.savefig(os.path.join(save_path, 'fuzzy_rules_consequents.png'))
    plt.close()

    print(f"总共生成 {active_rules} 条模糊规则。")
    print(f"规则后件可视化已保存到 {os.path.join(save_path, 'fuzzy_rules_consequents.png')}")


# ==========================================
# 主函数
# ==========================================
def main():
    set_seed(cfg.SEED)
    cfg.print_config()

    # [关键修改] 为此运行创建一个唯一的、带时间戳的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.MODEL_TYPE}_{cfg.EXTRACTOR_TYPE}_{timestamp}"
    SAVE_PATH = os.path.join('./checkpoints', run_name)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"所有结果将保存到: {SAVE_PATH}")
    # ---

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

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    best_model_save_path = os.path.join(SAVE_PATH, 'best_model.pth')

    print(f"\n开始训练 {cfg.MODEL_TYPE} 模型...")
    for epoch in range(cfg.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
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
            # [关键修改] 保存模型状态的同时，也保存 MAX_RULES 配置
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_rules': cfg.MAX_RULES
            }, best_model_save_path)
            print(f"    *** 新的最佳权重已保存至 {best_model_save_path} (Acc: {best_acc:.2f}%) ***")

        if cfg.MODEL_TYPE == 'DFM_FNCN':
            print(f"    当前规则总数: {model.classifier.num_active_rules.item()}/{cfg.MAX_RULES}\n")
        else:
            print("\n")

    print(f"训练结束! 最佳测试准确率: {best_acc:.2f}%")
    plot_history(history, SAVE_PATH)

    if cfg.MODEL_TYPE == 'DFM_FNCN':
        visualize_and_save_rules(model, SAVE_PATH)


if __name__ == '__main__':
    main()