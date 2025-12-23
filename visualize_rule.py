import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config as cfg
from models import FullModel
from train_decoder import get_decoder
import scipy.special


def configure_model_from_checkpoint(checkpoint):
    """从检查点恢复完整配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    if params['MODEL_TYPE'] != 'DFM_FNCN':
        print("错误: 只能可视化 DFM_FNCN 模型。")
        sys.exit()

    # 恢复基本配置
    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    # [关键修复] 恢复所有必要的配置参数
    if 'MAX_RULES' in params:
        cfg.MAX_RULES = params['MAX_RULES']

    if 'USE_ATTENTION' in params:
        cfg.USE_ATTENTION = params['USE_ATTENTION']

    if 'GTSRB_SUBSET_INDICES' in params:
        cfg.GTSRB_SUBSET_INDICES = params['GTSRB_SUBSET_INDICES']

    if 'INIT_SIGMA' in params:
        cfg.INIT_SIGMA = params['INIT_SIGMA']

    if 'PHI_TH' in params:
        cfg.PHI_TH = params['PHI_TH']

    # 根据恢复的配置更新数据集配置
    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")
    print(f"MAX_RULES={cfg.MAX_RULES}, USE_ATTENTION={cfg.USE_ATTENTION}")


def visualize_single_scale_rules(model, decoder, run_dir):
    """单尺度规则可视化（原始功能）"""
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    print(f"检测到 {num_rules} 条激活规则。")

    if num_rules == 0:
        print("没有激活的规则，无法可视化。")
        return

    centers = classifier.centers[:num_rules].detach()  # (Rules, 128, 36)
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()  # (Rules, Classes)

    # 获取注意力权重（如果存在）
    attention_weights = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        attention_weights = F.softmax(alpha, dim=1)  # (Rules, Channels)
        print(f"检测到注意力权重，形状: {attention_weights.shape}")

    # 将中心 reshape 为 (Rules, 128, 6, 6) 以输入解码器
    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 解码规则中心
    print("正在解码规则中心...")
    with torch.no_grad():
        # 根据解码器类型传递不同的参数
        if cfg.USE_ATTENTION_GUIDED_DECODER and attention_weights is not None:
            decoded_images = decoder(centers_reshaped, attention_weights)
            print("使用注意力引导解码")
        else:
            decoded_images = decoder(centers_reshaped)
            print("使用标准解码")

    # 反归一化图像 (假设 mean=0.5, std=0.5)
    decoded_images = decoded_images * 0.5 + 0.5
    decoded_images = torch.clamp(decoded_images, 0, 1)
    decoded_images = decoded_images.cpu().numpy()

    # 确定每条规则的预测类别
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 绘图
    print("正在生成可视化网格...")
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

        # 添加注意力信息（如果可用）
        title = f"R{i}: {class_name}\nConf: {confidences[i]:.2f}"
        if attention_weights is not None:
            # 获取该规则最重要的3个特征通道
            rule_attention = attention_weights[i].cpu().numpy()
            top_channels = np.argsort(rule_attention)[-3:][::-1]  # 降序排列
            title += f"\nTopCh: {top_channels.tolist()}"

        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(num_rules, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # 根据解码器类型选择保存路径
    if cfg.USE_ATTENTION_GUIDED_DECODER:
        save_path = os.path.join(run_dir, 'rules_visualized_attention_guided.png')
    else:
        save_path = os.path.join(run_dir, 'rules_visualized_labeled.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"单尺度可视化结果已保存至: {save_path}")

    # 可视化注意力权重热图
    if attention_weights is not None:
        print("正在生成注意力权重热图...")
        fig2, ax2 = plt.subplots(figsize=(12, max(6, num_rules * 0.3)))
        att_np = attention_weights.cpu().numpy()

        im = ax2.imshow(att_np, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Feature Channel Index')
        ax2.set_ylabel('Rule Index')
        ax2.set_title('Attention Weights per Rule (Softmax)')

        # 添加颜色条
        plt.colorbar(im, ax=ax2)

        # 设置y轴刻度
        ax2.set_yticks(range(num_rules))
        ax2.set_yticklabels([f"Rule {i}" for i in range(num_rules)])

        plt.tight_layout()
        attention_heatmap_path = os.path.join(run_dir, 'attention_weights_heatmap.png')
        plt.savefig(attention_heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"注意力权重热图已保存至: {attention_heatmap_path}")


def visualize_attention_guided_multi_scale_rules(model, decoder, run_dir):
    """注意力引导的多尺度规则可视化"""
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    if num_rules == 0:
        print("没有激活的规则，无法可视化。")
        return

    centers = classifier.centers[:num_rules].detach()
    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 获取注意力权重（如果存在）
    attention_weights = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        attention_weights = F.softmax(alpha, dim=1)
        print(f"检测到注意力权重，将用于多尺度解码")

    # 解码不同尺度
    print("正在解码注意力引导的多尺度规则中心...")
    with torch.no_grad():
        # 生成不同尺度的重建（注意力权重会被自动应用）
        coarse_images = decoder(centers_reshaped, attention_weights, scale='coarse')
        medium_images = decoder(centers_reshaped, attention_weights, scale='medium')
        fine_images = decoder(centers_reshaped, attention_weights, scale='fine')
        fused_images = decoder(centers_reshaped, attention_weights, scale='all')

    # 反归一化
    def denormalize(images):
        images = images * 0.5 + 0.5
        images = torch.clamp(images, 0, 1)
        return images.cpu().numpy()

    coarse_np = denormalize(coarse_images)
    medium_np = denormalize(medium_images)
    fine_np = denormalize(fine_images)
    fused_np = denormalize(fused_images)

    # 获取规则预测信息
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 创建注意力引导的多尺度可视化网格
    create_attention_guided_multi_scale_grid(
        coarse_np, medium_np, fine_np, fused_np,
        predicted_classes, confidences,
        attention_weights, run_dir, num_rules
    )

    # 创建尺度对比图（包含注意力信息）
    create_attention_scale_comparison(
        coarse_np, medium_np, fine_np, fused_np,
        predicted_classes, attention_weights, run_dir
    )


def create_attention_guided_multi_scale_grid(coarse, medium, fine, fused, pred_classes, confidences, attention_weights, run_dir, num_rules):
    """创建注意力引导的多尺度可视化网格"""
    cols = 4  # 4列：粗、中、细、融合
    rows = min(num_rules, 20)  # 最多显示20条规则

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(rows):
        # 粗尺度
        ax = axes[i, 0] if rows > 1 else axes[0]
        img = coarse[i].transpose(1, 2, 0) if coarse[i].shape[0] > 1 else coarse[i].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)

        # 添加注意力信息
        title = f"Rule {i}: Coarse\n{cfg.CLASS_NAMES[pred_classes[i]]}"
        if attention_weights is not None:
            rule_attention = attention_weights[i].cpu().numpy()
            top_channels = np.argsort(rule_attention)[-3:][::-1]
            title += f"\nTopCh: {top_channels.tolist()}"

        ax.set_title(title, fontsize=8)
        ax.axis('off')

        # 中尺度
        ax = axes[i, 1] if rows > 1 else axes[1]
        img = medium[i].transpose(1, 2, 0) if medium[i].shape[0] > 1 else medium[i].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        ax.set_title(f"Medium\nConf: {confidences[i]:.2f}", fontsize=8)
        ax.axis('off')

        # 细尺度
        ax = axes[i, 2] if rows > 1 else axes[2]
        img = fine[i].transpose(1, 2, 0) if fine[i].shape[0] > 1 else fine[i].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        ax.set_title("Fine", fontsize=8)
        ax.axis('off')

        # 融合
        ax = axes[i, 3] if rows > 1 else axes[3]
        img = fused[i].transpose(1, 2, 0) if fused[i].shape[0] > 1 else fused[i].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        ax.set_title("Fused (Att-Guided)", fontsize=8)
        ax.axis('off')

    plt.suptitle(f"Attention-Guided Multi-Scale Rule Visualization ({num_rules} Rules)", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(run_dir, 'rules_attention_guided_multi_scale.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"注意力引导的多尺度规则可视化已保存至: {save_path}")


def create_attention_scale_comparison(coarse, medium, fine, fused, pred_classes, attention_weights, run_dir):
    """创建尺度对比图（包含注意力信息）"""
    # 选择每个类别的代表性规则
    unique_classes = np.unique(pred_classes)
    selected_rules = []

    for cls in unique_classes[:5]:  # 最多5个类别
        cls_indices = np.where(pred_classes == cls)[0]
        if len(cls_indices) > 0:
            selected_rules.append(cls_indices[0])

    if len(selected_rules) == 0:
        return

    # 创建对比图
    fig, axes = plt.subplots(len(selected_rules), 4, figsize=(12, 3 * len(selected_rules)))

    for idx, rule_idx in enumerate(selected_rules):
        # 粗尺度
        ax = axes[idx, 0] if len(selected_rules) > 1 else axes[0]
        img = coarse[rule_idx].transpose(1, 2, 0) if coarse[rule_idx].shape[0] > 1 else coarse[rule_idx].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        if idx == 0:
            ax.set_title("Coarse Scale", fontsize=10)

        # 添加注意力信息
        ylabel = f"Rule {rule_idx}\n{cfg.CLASS_NAMES[pred_classes[rule_idx]]}"
        if attention_weights is not None:
            rule_attention = attention_weights[rule_idx].cpu().numpy()
            top_channels = np.argsort(rule_attention)[-2:][::-1]  # 显示最重要的2个通道
            ylabel += f"\nTopCh: {top_channels.tolist()}"

        ax.set_ylabel(ylabel, fontsize=9)
        ax.axis('off')

        # 中尺度
        ax = axes[idx, 1] if len(selected_rules) > 1 else axes[1]
        img = medium[rule_idx].transpose(1, 2, 0) if medium[rule_idx].shape[0] > 1 else medium[rule_idx].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        if idx == 0:
            ax.set_title("Medium Scale", fontsize=10)
        ax.axis('off')

        # 细尺度
        ax = axes[idx, 2] if len(selected_rules) > 1 else axes[2]
        img = fine[rule_idx].transpose(1, 2, 0) if fine[rule_idx].shape[0] > 1 else fine[rule_idx].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        if idx == 0:
            ax.set_title("Fine Scale", fontsize=10)
        ax.axis('off')

        # 融合
        ax = axes[idx, 3] if len(selected_rules) > 1 else axes[3]
        img = fused[rule_idx].transpose(1, 2, 0) if fused[rule_idx].shape[0] > 1 else fused[rule_idx].squeeze()
        ax.imshow(img, cmap='gray' if cfg.IN_CHANNELS == 1 else None)
        if idx == 0:
            ax.set_title("Fused (Att-Guided)", fontsize=10)
        ax.axis('off')

    plt.suptitle("Attention-Guided Multi-Scale Comparison (Representative Rules)", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(run_dir, 'attention_scale_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"注意力引导的尺度对比图已保存至: {save_path}")


def run_visualization(run_dir):
    """主可视化函数"""
    print(f"\n>>> 开始可视化规则 (Rule Visualization): {run_dir}")

    model_path = os.path.join(run_dir, 'best_model.pth')

    # 检查是否存在各种解码器类型
    attention_guided_multi_scale_path = os.path.join(run_dir, 'decoder_multi_scale.pth')
    attention_decoder_path = os.path.join(run_dir, 'decoder_attention_guided.pth')
    standard_decoder_path = os.path.join(run_dir, 'decoder.pth')

    # 根据存在的解码器文件决定使用哪个
    decoder_path = None
    decoder_type = None

    # 优先检查注意力引导的多尺度解码器
    if os.path.exists(attention_guided_multi_scale_path):
        decoder_path = attention_guided_multi_scale_path
        decoder_type = 'attention_guided_multi_scale'
        print("检测到注意力引导的多尺度解码器")
    elif os.path.exists(attention_decoder_path):
        decoder_path = attention_decoder_path
        decoder_type = 'attention_guided'
        print("检测到注意力引导解码器")
    elif os.path.exists(standard_decoder_path):
        decoder_path = standard_decoder_path
        decoder_type = 'standard'
        print("检测到标准解码器")
    else:
        print(f"错误: 找不到解码器文件。请检查 {run_dir}")
        return

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        return

    # 1. 加载模型和配置
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    configure_model_from_checkpoint(checkpoint)
    cfg.MAX_RULES = checkpoint['max_rules']

    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 根据解码器类型动态设置配置
    if decoder_type == 'attention_guided_multi_scale':
        # 强制使用注意力引导的多尺度解码器
        cfg.USE_MULTI_SCALE_VISUALIZATION = True
        cfg.USE_ATTENTION_GUIDED_DECODER = True
    elif decoder_type == 'attention_guided':
        # 强制使用注意力引导解码器
        cfg.USE_MULTI_SCALE_VISUALIZATION = False
        cfg.USE_ATTENTION_GUIDED_DECODER = True
    elif decoder_type == 'standard':
        # 使用标准解码器
        cfg.USE_MULTI_SCALE_VISUALIZATION = False
        cfg.USE_ATTENTION_GUIDED_DECODER = False

    # 3. 加载解码器
    decoder = get_decoder().to(cfg.DEVICE)
    decoder.load_state_dict(torch.load(decoder_path, map_location=cfg.DEVICE))
    decoder.eval()

    # 4. 根据解码器类型选择可视化函数
    if decoder_type == 'attention_guided_multi_scale':
        print("执行注意力引导的多尺度规则可视化...")
        visualize_attention_guided_multi_scale_rules(model, decoder, run_dir)
    elif decoder_type == 'attention_guided':
        print("执行注意力引导的单尺度规则可视化...")
        visualize_single_scale_rules(model, decoder, run_dir)
    else:
        print("执行标准单尺度规则可视化...")
        visualize_single_scale_rules(model, decoder, run_dir)

    print("规则可视化完成！")


if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = './checkpoints/！！！GTSRB_DFM_FNCN_RESNET18_PRETRAINED_20251209_115250'
    if os.path.exists(TEST_DIR):
        run_visualization(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")