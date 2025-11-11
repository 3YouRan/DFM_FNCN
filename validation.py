import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# 导入我们的自定义模块
import config as cfg
from models import FullModel

# --- 全局常量 ---
MODEL_PATH = os.path.join('./checkpoints', 'best_model.pth')
OUTPUT_DIR = './checkpoints'

# Fashion-MNIST 类别名称
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_model(model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        print("请先运行 train.py 训练并保存模型。")
        exit()

    print(f"正在从 {model_path} 加载模型...")
    # 1. 初始化模型结构
    model = FullModel().to(cfg.DEVICE)
    # 2. 加载保存的状态字典
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    # 3. 设置为评估模式 (关闭 dropout, batchnorm 更新等)
    model.eval()
    return model


def get_test_loader():
    """获取测试数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root=cfg.DATA_ROOT,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,  # 使用较大的 batch size 加快推理
        shuffle=False,
        num_workers=0
    )
    return test_loader


def run_inference(model, loader):
    """在测试集上运行推理并收集所有预测和标签"""
    print("正在测试集上运行推理...")
    all_preds = []
    all_targets = []

    with torch.no_grad():  # 推理时不需要计算梯度
        for data, targets in loader:
            data, targets = data.to(cfg.DEVICE), targets.to(cfg.DEVICE)

            # 运行模型
            outputs = model(data, labels=None, training_phase=False)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制并保存混淆矩阵"""
    print("正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',  # 整数格式
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16)
    plt.ylabel('真实类别 (True Label)', fontsize=12)
    plt.xlabel('预测类别 (Predicted Label)', fontsize=12)

    output_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵已保存至: {output_path}")
    plt.close()


def main():
    """主执行函数"""
    print("开始执行推理脚本...")

    # 1. 加载模型
    model = load_model(MODEL_PATH)

    # 2. 加载数据
    test_loader = get_test_loader()

    # 3. 运行推理
    predictions, targets = run_inference(model, test_loader)

    # 4. 生成分类报告
    print("\n" + "=" * 50)
    print("           分类报告 (Classification Report)")
    print("=" * 50)
    report = classification_report(targets, predictions, target_names=CLASS_NAMES)
    print(report)
    print("=" * 50 + "\n")

    # 5. 生成混淆矩阵图
    plot_confusion_matrix(targets, predictions, CLASS_NAMES, OUTPUT_DIR)

    print("推理完成。")


if __name__ == '__main__':
    main()