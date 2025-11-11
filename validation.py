import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import pandas as pd
import medmnist  # [新]
from medmnist import BloodMNIST  # [新]

# 导入我们的自定义模块
import config as cfg
from models import FullModel, TraditionalCNNModel

# --- [重要] 在此处配置您要评估的运行目录 ---
RUN_DIR_TO_VALIDATE = './checkpoints/YOUR_RUN_DIRECTORY_HERE'
# ---

MODEL_PATH = os.path.join(RUN_DIR_TO_VALIDATE, 'best_model.pth')
OUTPUT_DIR = RUN_DIR_TO_VALIDATE


def configure_model_from_checkpoint(checkpoint):
    """[新] 从检查点中的 'config_params' 推断配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。请从目录名中手动推断配置。")
        # 您可以保留旧的推断逻辑作为备用
        sys.exit()

    params = checkpoint['config_params']
    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    # [新] 加载数据集特定的配置
    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")


def load_model_and_config(model_path):
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        sys.exit()

    print(f"正在从 {model_path} 加载模型...")

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)

    # 1. [新] 自动配置
    configure_model_from_checkpoint(checkpoint)

    # 2. 从检查点中恢复 MAX_RULES
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        if 'max_rules' not in checkpoint:
            print("错误: 模糊模型检查点不包含 'max_rules'。")
            sys.exit()
        cfg.MAX_RULES = checkpoint['max_rules']
        print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")

    # 3. 构建模型骨架
    if cfg.MODEL_TYPE == 'DFM_FNCN':
        model = FullModel().to(cfg.DEVICE)
    else:
        model = TraditionalCNNModel().to(cfg.DEVICE)

    # 4. 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_test_loader():
    """[新] 根据 config.py 动态加载数据集"""
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
        test_dataset = datasets.FashionMNIST(
            root=cfg.DATA_ROOT, train=False, download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'SVHN':
        test_dataset = datasets.SVHN(
            root=cfg.DATA_ROOT, split='test', download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        test_dataset = BloodMNIST(
            split='test', transform=data_transform, download=True, root=cfg.DATA_ROOT
        )
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    return test_loader


def run_inference(model, loader):
    print("正在测试集上运行推理...")
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data_tuple in loader:
            data, targets = data_tuple[0].to(cfg.DEVICE), data_tuple[1].to(cfg.DEVICE)
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            if cfg.MODEL_TYPE == 'DFM_FNCN':
                outputs = model(data, labels=None, training_phase=False)
            else:
                outputs = model(data)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    print("正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16)
    plt.ylabel('真实类别 (True Label)', fontsize=12)
    plt.xlabel('预测类别 (Predicted Label)', fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(output_path)
    print(f"混淆矩阵已保存至: {output_path}")
    plt.close()


def main():
    print("开始执行推理脚本...")

    # 1. 加载模型 (这将自动配置 cfg)
    model = load_model_and_config(MODEL_PATH)

    # 2. 加载数据
    test_loader = get_test_loader()

    # 3. 运行推理
    predictions, targets = run_inference(model, test_loader)

    # 4. 生成分类报告 (打印)
    print("\n" + "=" * 50)
    print("           分类报告 (Classification Report)")
    print("=" * 50)
    report_str = classification_report(targets, predictions, target_names=cfg.CLASS_NAMES, zero_division=0)
    print(report_str)
    print("=" * 50 + "\n")

    # 5. 生成 CSV
    print("正在将分类报告保存到 CSV...")
    report_dict = classification_report(targets, predictions, target_names=cfg.CLASS_NAMES, zero_division=0,
                                        output_dict=True)
    accuracy_value = report_dict.pop('accuracy')
    df = pd.DataFrame(report_dict).transpose()
    df.loc['accuracy'] = pd.Series(
        {'f1-score': accuracy_value, 'support': df.loc['weighted avg', 'support']}
    )
    csv_save_path = os.path.join(OUTPUT_DIR, 'classification_report.csv')
    df.to_csv(csv_save_path, index=True, encoding='utf-8-sig')
    print(f"分类报告已保存至: {csv_save_path}")

    # 6. 生成混淆矩阵图
    plot_confusion_matrix(targets, predictions, cfg.CLASS_NAMES, OUTPUT_DIR)

    print(f"\n推理完成。所有报告已保存到 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()