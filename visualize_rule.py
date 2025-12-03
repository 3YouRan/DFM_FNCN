import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config as cfg
from models import FullModel
from train_decoder import get_decoder
import scipy.special


# 移除硬编码的 RUN_DIR_TO_VISUALIZE

def configure_model_from_checkpoint(checkpoint):
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    if params['MODEL_TYPE'] != 'DFM_FNCN':
        print("错误: 只能可视化 DFM_FNCN 模型。")
        sys.exit()

    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    # [创新点1] 恢复 Attention 配置
    if 'USE_ATTENTION' in params:
        cfg.USE_ATTENTION = params['USE_ATTENTION']

    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")


def run_visualization(run_dir):
    """[修改] 接收 run_dir 参数供 main.py 调用"""
    print(f"\n>>> 开始可视化规则 (Rule Visualization): {run_dir}")

    model_path = os.path.join(run_dir, 'best_model.pth')
    decoder_path = os.path.join(run_dir, 'decoder.pth')
    save_path = os.path.join(run_dir, 'rules_visualized_labeled.png')

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        return
    if not os.path.exists(decoder_path):
        print(f"错误: 找不到解码器文件 '{decoder_path}'。请先运行 train_decoder.py")
        return

    # 1. 加载模型和配置
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    configure_model_from_checkpoint(checkpoint)
    cfg.MAX_RULES = checkpoint['max_rules']

    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 加载解码器
    decoder = get_decoder().to(cfg.DEVICE)
    decoder.load_state_dict(torch.load(decoder_path, map_location=cfg.DEVICE))
    decoder.eval()

    # 3. 获取规则中心和后件
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    print(f"检测到 {num_rules} 条激活规则。")

    if num_rules == 0:
        print("没有激活的规则，无法可视化。")
        return

    centers = classifier.centers[:num_rules].detach()  # (Rules, 128, 36)
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()  # (Rules, Classes)

    # 将中心 reshape 为 (Rules, 128, 6, 6) 以输入解码器
    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 4. 解码规则中心
    print("正在解码规则中心...")
    with torch.no_grad():
        decoded_images = decoder(centers_reshaped)

    # 反归一化图像 (假设 mean=0.5, std=0.5)
    decoded_images = decoded_images * 0.5 + 0.5
    decoded_images = torch.clamp(decoded_images, 0, 1)
    decoded_images = decoded_images.cpu().numpy()

    # 5. 确定每条规则的预测类别
    # 使用 Softmax 找到概率最高的类
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 6. 绘图
    print("正在生成可视化网格...")
    # 计算网格尺寸
    cols = 10
    rows = (num_rules + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.5 * rows))
    axes = axes.flatten()

    for i in range(num_rules):
        ax = axes[i]
        img = decoded_images[i]

        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        if cfg.IN_CHANNELS == 1:
            img = img.squeeze()
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        class_name = cfg.CLASS_NAMES[predicted_classes[i]]
        ax.set_title(f"R{i}: {class_name}\nConf: {confidences[i]:.2f}", fontsize=9)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(num_rules, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"可视化结果已保存至: {save_path}")


if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = './checkpoints/YOUR_RUN_DIR'
    if os.path.exists(TEST_DIR):
        run_visualization(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")