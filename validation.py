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
import sys
import pandas as pd  # [新] 导入 pandas 库

# 导入我们的自定义模块
import config as cfg
from models import FullModel, TraditionalCNNModel

# --- [重要] 在此处配置您要评估的运行目录 ---
# --- 例如: './checkpoints/DFM_FNCN_RESNET18_PRETRAINED_20251111_1830'
RUN_DIR_TO_VALIDATE = './checkpoints/DFM_FNCN_VGG16_PRETRAINED_20251111_180114'
# ---

MODEL_PATH = os.path.join(RUN_DIR_TO_VALIDATE, 'best_model.pth')
OUTPUT_DIR = RUN_DIR_TO_VALIDATE  # 将报告保存在同一目录

CLASS_NAMES = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def configure_model_from_path(run_dir):
    """
    智能地从目录名称中推断模型配置并设置 config.py
    """
    if not os.path.exists(run_dir):
        print(f"错误: 目录不存在 '{run_dir}'")
        print("请确保 RUN_DIR_TO_VALIDATE 指向一个已存在的训练目录。")
        sys.exit()

    print(f"正在从目录名中推断配置: {run_dir}")
    base_name = os.path.basename(run_dir)

    if 'DFM_FNCN' in base_name:
        cfg.MODEL_TYPE = 'DFM_FNCN'
    elif 'TRADITIONAL_CNN' in base_name:
        cfg.MODEL_TYPE = 'TRADITIONAL_CNN'
    else:
        raise ValueError(f"无法从目录名中推断 MODEL_TYPE。'{base_name}'")

    if 'RESNET18_PRETRAINED' in base_name:
        cfg.EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'
    elif 'VGG16_PRETRAINED' in base_name:
        cfg.EXTRACTOR_TYPE = 'VGG16_PRETRAINED'
    elif 'SIMPLE_CNN' in base_name:
        cfg.EXTRACTOR_TYPE = 'SIMPLE_CNN'
    else:
        raise ValueError(f"无法从目录名中推断 EXTRACTOR_TYPE。'{base_name}'")

    print(f"推断配置: MODEL_TYPE={cfg.MODEL_TYPE}, EXTRACTOR_TYPE={cfg.EXTRACTOR_TYPE}")


def load_model(model_path):
    """
    根据推断出的配置加载正确的模型
    """
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        print("请确保 RUN_DIR_TO_VALIDATE 设置正确，并已成功训练。")
        sys.exit()

    print(f"正在从 {model_path} 加载模型...")

    # 1. 加载完整的检查点文件
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)

    # 2. 从检查点中恢复 MAX_RULES 并更新配置
    if 'max_rules' in checkpoint:
        cfg.MAX_RULES = checkpoint['max_rules']
        print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")
    elif cfg.MODEL_TYPE == 'DFM_FNCN':
        print("警告: 检查点不包含 'max_rules'。将使用 config.py 中的默认值。")
        # 允许加载旧的、不包含 'max_rules' 的模型（如果尺寸凑巧一样）

    # 3. 根据推断出的配置构建正确的模型骨架
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        model = FullModel().to(cfg.DEVICE)
    else:
        model = TraditionalCNNModel().to(cfg.DEVICE)

    # 4. 加载权重
    # 检查 state_dict 是在 'model_state_dict' 键下还是直接保存的
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 兼容旧的保存格式

    model.eval()
    return model


def get_test_loader():
    """获取测试数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    return test_loader


def run_inference(model, loader):
    """
    在推理时使用正确的 forward 调用
    """
    print("正在测试集上运行推理...")
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(cfg.DEVICE), targets.to(cfg.DEVICE)

            if cfg.MODEL_TYPE == 'DFM_FNCN':
                outputs = model(data, labels=None, training_phase=False)
            else:  # 'TRADITIONAL_CNN'
                outputs = model(data)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制并保存混淆矩阵"""
    print("正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
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

    # 1. 智能配置
    configure_model_from_path(RUN_DIR_TO_VALIDATE)

    # 2. 加载模型
    model = load_model(MODEL_PATH)

    # 3. 加载数据
    test_loader = get_test_loader()

    # 4. 运行推理
    predictions, targets = run_inference(model, test_loader)

    # 5. 生成分类报告 (用于打印)
    print("\n" + "=" * 50)
    print("           分类报告 (Classification Report)")
    print("=" * 50)
    report_str = classification_report(targets, predictions, target_names=CLASS_NAMES, zero_division=0)
    print(report_str)
    print("=" * 50 + "\n")

    # 6. [新] 生成分类报告 (用于保存 CSV)
    print("正在将分类报告保存到 CSV...")
    report_dict = classification_report(targets, predictions, target_names=CLASS_NAMES, zero_division=0,
                                        output_dict=True)

    # 从字典中提取 accuracy，因为它不是一个嵌套字典
    try:
        accuracy_value = report_dict.pop('accuracy')
    except KeyError:
        accuracy_value = None

    # 将剩余的字典 (每个类 + avg) 转换为 DataFrame
    df = pd.DataFrame(report_dict).transpose()

    # 如果提取到了 accuracy，将其作为一个新行添加回去，以便在 CSV 中查看
    if accuracy_value is not None:
        df.loc['accuracy'] = pd.Series(
            {'f1-score': accuracy_value, 'support': df.loc['weighted avg', 'support']}
        )

    # 定义保存路径
    csv_save_path = os.path.join(OUTPUT_DIR, 'classification_report.csv')

    # 保存为 CSV (使用 utf-8-sig 编码以确保 Excel 正确读取)
    df.to_csv(csv_save_path, index=True, encoding='utf-8-sig')
    print(f"分类报告已保存至: {csv_save_path}")

    # 7. 生成混淆矩阵图
    plot_confusion_matrix(targets, predictions, CLASS_NAMES, OUTPUT_DIR)

    print(f"\n推理完成。所有报告已保存到 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()