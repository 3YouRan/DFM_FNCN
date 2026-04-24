import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import scipy.special
import pandas as pd
from datetime import datetime
import warnings
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# 图像质量评估相关导入
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not found, PSNR and SSIM metrics will not be available")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    warnings.warn("lpips not found, LPIPS metric will not be available")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False
    warnings.warn("torchmetrics not found, FID metric will not be available")

import config as cfg
from models import FullModel
from train_decoder import (
    SimpleCNNDecoder, ResNet18Decoder, VGG16Decoder,
    AttentionGuidedDecoder, MultiScaleDecoder, AttentionGuidedMultiScaleDecoder,
    GANDecoder
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def configure_model_from_checkpoint(checkpoint):
    """从检查点恢复配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    # [创新点1] 恢复 Attention 配置
    if 'USE_ATTENTION' in params:
        cfg.USE_ATTENTION = params['USE_ATTENTION']

    # [创新点4] 恢复解码器配置
    if 'USE_ATTENTION_GUIDED_DECODER' in params:
        cfg.USE_ATTENTION_GUIDED_DECODER = params['USE_ATTENTION_GUIDED_DECODER']
    if 'USE_MULTI_SCALE_VISUALIZATION' in params:
        cfg.USE_MULTI_SCALE_VISUALIZATION = params['USE_MULTI_SCALE_VISUALIZATION']
    if 'USE_GAN_DECODER' in params:
        cfg.USE_GAN_DECODER = params['USE_GAN_DECODER']

    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']
    
    # 设置 IMG_DIM_OUT 和 P_DIM
    if cfg.DATASET_NAME == 'VEHICLES':
        cfg.IMG_DIM_OUT = 15
    elif cfg.DATASET_NAME == 'FASHION_MNIST' or cfg.DATASET_NAME == 'MNIST' or cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION' or cfg.DATASET_NAME == 'GTSRB':
        cfg.IMG_DIM_OUT = 6
    else:
        cfg.IMG_DIM_OUT = 7
    cfg.P_DIM = cfg.IMG_DIM_OUT * cfg.IMG_DIM_OUT
    cfg.N_CHANNELS_OUT = params.get('N_CHANNELS_OUT', 128)
    
    # 从检查点恢复 MAX_RULES（如果有的话）
    # 优先从 config_params 获取，如果没有则使用默认值
    if 'max_rules' in checkpoint:
        cfg.MAX_RULES = checkpoint['max_rules']
    elif 'MAX_RULES' in params:
        cfg.MAX_RULES = params['MAX_RULES']
    else:
        cfg.MAX_RULES = 200
    
    print(f"MAX_RULES: {cfg.MAX_RULES}")

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")
    print(f"ATTENTION: {cfg.USE_ATTENTION}, ATTENTION_GUIDED_DECODER: {cfg.USE_ATTENTION_GUIDED_DECODER}")
    print(f"MULTI_SCALE: {cfg.USE_MULTI_SCALE_VISUALIZATION}, GAN_DECODER: {cfg.USE_GAN_DECODER}")
    print(f"IMG_DIM_OUT: {cfg.IMG_DIM_OUT}, N_CHANNELS_OUT: {cfg.N_CHANNELS_OUT}")


def get_decoder_from_config(run_dir, decoder_type='both'):
    """根据配置选择并加载解码器
    
    Args:
        run_dir: 运行目录
        decoder_type: 'standard' - 标准解码器, 'gan' - GAN解码器, 'both' - 返回两种解码器
    """
    # 尝试加载新的解码器信息文件格式
    decoder_info_path = os.path.join(run_dir, 'decoder_info.pth')
    
    # 默认路径
    standard_decoder_path = os.path.join(run_dir, 'decoder_standard.pth')
    gan_decoder_path = os.path.join(run_dir, 'decoder_gan.pth')
    gan_discriminator_path = os.path.join(run_dir, 'discriminator.pth')
    
    decoder_info = None
    use_new_format = False
    
    if os.path.exists(decoder_info_path):
        try:
            decoder_info = torch.load(decoder_info_path, map_location=cfg.DEVICE, weights_only=False)
            print(f"加载解码器信息: {decoder_info}")
            
            # 检查是否为新格式
            if 'standard_decoder_path' in decoder_info:
                use_new_format = True
                standard_decoder_path = decoder_info.get('standard_decoder_path', standard_decoder_path)
                gan_decoder_path = decoder_info.get('gan_decoder_path', gan_decoder_path)
                gan_discriminator_path = decoder_info.get('gan_discriminator_path', gan_discriminator_path)
        except Exception as e:
            print(f"加载解码器信息失败: {e}")
    
    # 确定基础解码器类
    if cfg.EXTRACTOR_TYPE == 'RESNET18_PRETRAINED':
        base_class = ResNet18Decoder
    elif cfg.EXTRACTOR_TYPE == 'VGG16_PRETRAINED':
        base_class = VGG16Decoder
    elif cfg.EXTRACTOR_TYPE == 'SIMPLE_CNN':
        base_class = SimpleCNNDecoder
    else:
        raise ValueError(f"未知的 EXTRACTOR_TYPE: {cfg.EXTRACTOR_TYPE}")
    
    decoders = {}
    
    # 加载标准解码器
    if decoder_type in ['standard', 'both']:
        if os.path.exists(standard_decoder_path):
            print(f"加载标准解码器: {standard_decoder_path}")
            
            # 检查保存的权重是否包含注意力引导的键
            try:
                state_dict = torch.load(standard_decoder_path, map_location=cfg.DEVICE, weights_only=True)
                has_attention = any('attention' in k for k in state_dict.keys())
                has_base_decoder = any('base_decoder' in k for k in state_dict.keys())
            except:
                has_attention = False
                has_base_decoder = False
            
            # 根据权重类型选择正确的解码器
            if has_attention or has_base_decoder:
                print("  检测到注意力引导解码器权重，使用 AttentionGuidedDecoder")
                standard_decoder = AttentionGuidedDecoder(base_class).to(cfg.DEVICE)
            else:
                standard_decoder = base_class().to(cfg.DEVICE)
            
            # 使用 strict=False 加载
            standard_decoder.load_state_dict(torch.load(standard_decoder_path, map_location=cfg.DEVICE, weights_only=False), strict=False)
            standard_decoder.eval()
            decoders['standard'] = standard_decoder
        else:
            print(f"警告: 标准解码器文件不存在: {standard_decoder_path}")
    
    # 加载GAN解码器
    if decoder_type in ['gan', 'both']:
        if os.path.exists(gan_decoder_path):
            print(f"加载GAN解码器: {gan_decoder_path}")
            gan_decoder = GANDecoder(base_class).to(cfg.DEVICE)
            gan_decoder.generator.load_state_dict(torch.load(gan_decoder_path, map_location=cfg.DEVICE, weights_only=False))
            
            if os.path.exists(gan_discriminator_path):
                gan_decoder.discriminator.load_state_dict(torch.load(gan_discriminator_path, map_location=cfg.DEVICE, weights_only=False))
                print(f"加载判别器: {gan_discriminator_path}")
            
            gan_decoder.eval()
            decoders['gan'] = gan_decoder
        else:
            print(f"警告: GAN解码器文件不存在: {gan_decoder_path}")
    
    # 如果只请求一种解码器，直接返回
    if decoder_type in ['standard', 'gan']:
        if decoder_type in decoders:
            return decoders[decoder_type]
        else:
            print(f"错误: 找不到 {decoder_type} 解码器")
            return None
    
    # 返回所有加载的解码器
    return decoders

def decode_rule_centers(decoder, centers_reshaped, rule_specific_attention=None):
    """统一的规则中心解码函数"""
    decoded_images = []
    num_rules = centers_reshaped.size(0)

    with torch.no_grad():
        for i in range(num_rules):
            rule_center = centers_reshaped[i:i + 1]  # (1, C, H, W)

            if rule_specific_attention is not None:
                # 使用规则特定的注意力权重
                rule_attention = rule_specific_attention[i:i + 1]  # (1, C)

                # 根据解码器类型调用不同的前向传播
                if isinstance(decoder, (AttentionGuidedDecoder, AttentionGuidedMultiScaleDecoder)):
                    decoded = decoder(rule_center, rule_attention)
                elif isinstance(decoder, MultiScaleDecoder):
                    decoded = decoder(rule_center, rule_attention, scale='all')
                elif isinstance(decoder, GANDecoder):
                    # GAN解码器可能支持注意力权重（如果基础解码器支持）
                    decoded = decoder(rule_center, rule_attention)
                else:
                    decoded = decoder(rule_center)
            else:
                # 没有注意力权重
                if isinstance(decoder, MultiScaleDecoder):
                    decoded = decoder(rule_center, scale='all')
                elif isinstance(decoder, GANDecoder):
                    # GAN解码器前向传播
                    decoded = decoder(rule_center)
                else:
                    decoded = decoder(rule_center)

            decoded_images.append(decoded)

    # 合并所有解码结果
    decoded_images = torch.cat(decoded_images, dim=0)

    # 反归一化图像 (假设 mean=0.5, std=0.5)
    decoded_images = decoded_images * 0.5 + 0.5
    decoded_images = torch.clamp(decoded_images, 0, 1)

    return decoded_images.cpu().numpy()


def visualize_single_scale_rules(model, decoder, run_dir, decoder_type='standard'):
    """单尺度规则可视化（支持标准解码器和GAN解码器）"""
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    print(f"使用 {decoder_type} 解码器，检测到 {num_rules} 条激活规则。")

    if num_rules == 0:
        print("没有激活的规则，无法可视化。")
        return

    centers = classifier.centers[:num_rules].detach()  # (Rules, 128, 36)
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()  # (Rules, Classes)

    # 获取每条规则自己的注意力权重
    rule_specific_attention = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        rule_specific_attention = F.softmax(alpha, dim=1)  # (Rules, Channels)
        print(f"使用规则特定的注意力权重，形状: {rule_specific_attention.shape}")

    # 将中心 reshape 为 (Rules, 128, 6, 6) 以输入解码器
    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 解码规则中心
    print(f"正在使用 {decoder_type} 解码器解码规则中心...")
    decoded_images = decode_rule_centers(decoder, centers_reshaped, rule_specific_attention)

    # 确定每条规则的预测类别
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 绘图
    print("正在生成可视化网格...")

    # 动态调整列数和子图大小
    if num_rules <= 10:
        cols = 5
        fig_width = 20
        fig_height_per_row = 4.5
    elif num_rules <= 20:
        cols = 6
        fig_width = 24
        fig_height_per_row = 4.0
    else:
        cols = 8
        fig_width = 28
        fig_height_per_row = 3.5

    rows = (num_rules + cols - 1) // cols
    fig_height = fig_height_per_row * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

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

        title_lines = []
        title_lines.append(f"R{i}: {class_name}")
        title_lines.append(f"Conf: {confidences[i]:.2f}")

        if rule_specific_attention is not None:
            rule_attention = rule_specific_attention[i].cpu().numpy()
            top_channels = np.argsort(rule_attention)[-3:][::-1]
            top_weights = rule_attention[top_channels]

            if len(top_channels) > 0:
                title_lines.append(f"TopCh: {top_channels[0]}")
                if len(top_channels) > 1:
                    title_lines.append(f"W: {top_weights[0]:.2f},{top_weights[1]:.2f}")

        title = "\n".join(title_lines)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis('off')

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#888888')
            spine.set_linewidth(0.5)

    # 隐藏多余的子图
    for i in range(num_rules, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)

    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=1.5)

    # 根据解码器类型选择保存路径
    if decoder_type == 'gan':
        save_path = os.path.join(run_dir, 'rules_visualized_gan.png')
    else:
        save_path = os.path.join(run_dir, 'rules_visualized_standard.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{decoder_type} 规则可视化结果已保存至: {save_path}")

    return save_path


def visualize_comparison_rules(model, decoders, run_dir):
    """对比可视化：同时显示标准解码器和GAN解码器的结果"""
    if not decoders or len(decoders) < 2:
        print("没有足够的解码器进行对比可视化")
        return
    
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    print(f"\n进行解码器对比可视化，检测到 {num_rules} 条激活规则...")

    if num_rules == 0:
        print("没有激活的规则，无法可视化。")
        return

    centers = classifier.centers[:num_rules].detach()
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()

    # 获取注意力权重
    rule_specific_attention = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        rule_specific_attention = F.softmax(alpha, dim=1)

    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 解码所有规则
    decoded_results = {}
    for decoder_type, decoder in decoders.items():
        print(f"  正在解码: {decoder_type}")
        decoded_images = decode_rule_centers(decoder, centers_reshaped, rule_specific_attention)
        decoded_results[decoder_type] = decoded_images

    # 确定每条规则的预测类别
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 创建对比图：每行显示一个规则的两种解码结果
    print("正在生成对比可视化...")
    
    cols = len(decoders) + 1  # +1 用于类别标签
    rows = num_rules
    
    fig_width = cols * 4
    fig_height = rows * 3.5
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(num_rules):
        ax_idx = i * cols
        
        # 第一列显示规则信息
        ax = axes[ax_idx]
        ax.axis('off')
        class_name = cfg.CLASS_NAMES[predicted_classes[i]]
        info_text = f"Rule {i}\n{class_name}\nConf: {confidences[i]:.2f}"
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes, 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 其他列显示不同解码器的结果
        for j, (decoder_type, decoded_images) in enumerate(decoded_results.items()):
            ax = axes[ax_idx + j + 1]
            img = decoded_images[i]
            img = np.transpose(img, (1, 2, 0))
            
            if cfg.IN_CHANNELS == 1:
                img = img.squeeze()
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            ax.set_title(f"{decoder_type}", fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(run_dir, 'rules_visualized_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比可视化已保存至: {save_path}")
    
    return save_path


def visualize_attention_heatmap(classifier, num_rules, run_dir):
    """可视化注意力权重热图"""
    rule_specific_attention = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        rule_specific_attention = F.softmax(alpha, dim=1)
    
    if rule_specific_attention is None:
        return
    
    print("正在生成规则特定注意力权重热图...")
    fig2, ax2 = plt.subplots(figsize=(14, max(8, num_rules * 0.4)))
    att_np = rule_specific_attention.cpu().numpy()

    im = ax2.imshow(att_np, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Feature Channel Index', fontsize=10)
    ax2.set_ylabel('Rule Index', fontsize=10)
    ax2.set_title('Rule-Specific Attention Weights (Softmax)', fontsize=12, pad=15)

    # 设置x轴刻度 - 每10个通道显示一个刻度
    x_ticks = np.arange(0, att_np.shape[1], 10)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticks, fontsize=8)

    # 设置y轴刻度
    y_ticks = np.arange(0, num_rules, max(1, num_rules // 20))
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f"R{i}" for i in y_ticks], fontsize=8)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    attention_heatmap_path = os.path.join(run_dir, 'rule_specific_attention_weights_heatmap.png')
    plt.savefig(attention_heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"规则特定注意力权重热图已保存至: {attention_heatmap_path}")


def visualize_multi_scale_rules(model, decoder, run_dir):
    """多尺度规则可视化（如果启用多尺度解码器）"""
    if not cfg.USE_MULTI_SCALE_VISUALIZATION:
        return

    # 确定实际用于多尺度解码的生成器
    if isinstance(decoder, GANDecoder):
        print("检测到GAN解码器，使用其生成器进行多尺度可视化")
        generator = decoder.generator
        # 检查生成器是否为多尺度解码器
        if not isinstance(generator, (MultiScaleDecoder, AttentionGuidedMultiScaleDecoder)):
            print("警告: GAN解码器的生成器不是多尺度解码器，跳过多尺度可视化")
            return
        scale_decoder = generator
    elif isinstance(decoder, (MultiScaleDecoder, AttentionGuidedMultiScaleDecoder)):
        scale_decoder = decoder
    else:
        print("警告: 解码器不是多尺度解码器，跳过多尺度可视化")
        return

    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    print(f"检测到 {num_rules} 条激活规则，进行多尺度可视化...")

    if num_rules == 0:
        return

    centers = classifier.centers[:num_rules].detach()
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()

    # 获取注意力权重
    rule_specific_attention = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        rule_specific_attention = F.softmax(alpha, dim=1)

    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    # 确定每条规则的预测类别
    consequents_prob = scipy.special.softmax(consequents, axis=1)
    predicted_classes = np.argmax(consequents_prob, axis=1)
    confidences = np.max(consequents_prob, axis=1)

    # 为每个尺度生成可视化
    scales = ['coarse', 'medium', 'fine', 'all']
    scale_names = ['粗尺度', '中尺度', '细尺度', '融合结果']

    for scale_idx, (scale, scale_name) in enumerate(zip(scales, scale_names)):
        print(f"正在生成 {scale_name} 可视化...")

        decoded_images = []
        with torch.no_grad():
            for i in range(num_rules):
                rule_center = centers_reshaped[i:i + 1]

                if rule_specific_attention is not None:
                    rule_attention = rule_specific_attention[i:i + 1]
                    if isinstance(scale_decoder, (MultiScaleDecoder, AttentionGuidedMultiScaleDecoder)):
                        decoded = scale_decoder(rule_center, rule_attention, scale=scale)
                    else:
                        decoded = scale_decoder(rule_center, rule_attention)
                else:
                    if isinstance(scale_decoder, (MultiScaleDecoder, AttentionGuidedMultiScaleDecoder)):
                        decoded = scale_decoder(rule_center, scale=scale)
                    else:
                        decoded = scale_decoder(rule_center)

                decoded_images.append(decoded)

        decoded_images = torch.cat(decoded_images, dim=0)
        decoded_images = decoded_images * 0.5 + 0.5
        decoded_images = torch.clamp(decoded_images, 0, 1)
        decoded_images = decoded_images.cpu().numpy()

        # 绘制该尺度的可视化
        cols = min(8, num_rules)
        rows = (num_rules + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))

        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        for i in range(num_rules):
            ax = axes[i]
            img = decoded_images[i]
            img = np.transpose(img, (1, 2, 0))

            if cfg.IN_CHANNELS == 1:
                img = img.squeeze()
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

            class_name = cfg.CLASS_NAMES[predicted_classes[i]]
            ax.set_title(f"R{i}: {class_name}\nConf: {confidences[i]:.2f}", fontsize=8)
            ax.axis('off')

        for i in range(num_rules, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)

        plt.suptitle(f'{scale_name} 规则可视化', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(run_dir, f'rules_visualized_{scale}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"{scale_name} 可视化已保存至: {save_path}")


def create_rule_summary_table(run_dir, predicted_classes, confidences, rule_specific_attention):
    """创建规则信息摘要表格"""
    if rule_specific_attention is None:
        return

    num_rules = len(predicted_classes)

    # 收集规则信息
    rule_data = []
    for i in range(num_rules):
        rule_info = {
            'Rule': i,
            'Class': cfg.CLASS_NAMES[predicted_classes[i]],
            'Confidence': f"{confidences[i]:.3f}"
        }

        # 添加注意力信息
        rule_attention = rule_specific_attention[i].cpu().numpy()
        top_channels = np.argsort(rule_attention)[-3:][::-1]
        top_weights = rule_attention[top_channels]

        for j in range(min(3, len(top_channels))):
            rule_info[f'TopCh{j + 1}'] = top_channels[j]
            rule_info[f'W{j + 1}'] = f"{top_weights[j]:.3f}"

        rule_data.append(rule_info)

    # 创建DataFrame
    df = pd.DataFrame(rule_data)

    # 保存为CSV
    csv_path = os.path.join(run_dir, 'rule_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"规则摘要表格已保存至: {csv_path}")

    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(12, max(6, num_rules * 0.3)))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格
    table_data = []
    headers = ['Rule', 'Class', 'Confidence', 'TopCh1', 'W1', 'TopCh2', 'W2', 'TopCh3', 'W3']

    for i in range(num_rules):
        row = [
            f"R{i}",
            cfg.CLASS_NAMES[predicted_classes[i]],
            f"{confidences[i]:.3f}"
        ]

        rule_attention = rule_specific_attention[i].cpu().numpy()
        top_channels = np.argsort(rule_attention)[-3:][::-1]
        top_weights = rule_attention[top_channels]

        for j in range(3):
            if j < len(top_channels):
                row.append(str(top_channels[j]))
                row.append(f"{top_weights[j]:.3f}")
            else:
                row.append('')
                row.append('')

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Rule Summary Table', fontsize=14, pad=20)
    plt.tight_layout()

    table_path = os.path.join(run_dir, 'rule_summary_table.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"规则摘要表格图像已保存至: {table_path}")


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """计算PSNR（峰值信噪比）"""
    if not HAS_SKIMAGE:
        return float('nan')
    
    try:
        if img1.shape != img2.shape:
            from skimage.transform import resize
            img2 = resize(img2, img1.shape, preserve_range=True)
        return psnr(img1, img2, data_range=data_range)
    except Exception as e:
        print(f"PSNR计算错误: {e}")
        return float('nan')


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """计算SSIM（结构相似性指数）"""
    if not HAS_SKIMAGE:
        return float('nan')
    
    try:
        if img1.shape != img2.shape:
            from skimage.transform import resize
            img2 = resize(img2, img1.shape, preserve_range=True)
        
        if len(img1.shape) == 2:
            return ssim(img1, img2, data_range=data_range)
        else:
            channel_axis = 2 if img1.shape[-1] <= 3 else 0
            return ssim(img1, img2, data_range=data_range, channel_axis=channel_axis)
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return float('nan')


def calculate_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算LPIPS（学习感知图像块相似度）"""
    if not HAS_LPIPS:
        return float('nan')
    
    try:
        # 确保图像尺寸足够大（LPIPS需要至少64x64）
        min_size = 64
        
        # 调整图像尺寸
        from skimage.transform import resize
        
        # 获取当前尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 如果图像太小，调整到最小尺寸
        if h1 < min_size or w1 < min_size:
            scale = max(min_size / h1, min_size / w1)
            new_h, new_w = int(h1 * scale), int(w1 * scale)
            img1 = resize(img1, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)
        
        if h2 < min_size or w2 < min_size:
            scale = max(min_size / h2, min_size / w2)
            new_h, new_w = int(h2 * scale), int(w2 * scale)
            img2 = resize(img2, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)
        
        # 确保两个图像尺寸相同
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape, order=1, preserve_range=True, anti_aliasing=True)
        
        loss_fn = lpips.LPIPS(net='alex', verbose=False).to(cfg.DEVICE)
        
        # 准备图像张量
        def prepare_tensor(img):
            if len(img.shape) == 3 and img.shape[0] == 3:
                tensor = torch.from_numpy(img).float().unsqueeze(0)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            else:
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=0)
                tensor = torch.from_numpy(img).float().unsqueeze(0)
            return tensor * 2 - 1  # 归一化到[-1, 1]
        
        tensor1 = prepare_tensor(img1).to(cfg.DEVICE)
        tensor2 = prepare_tensor(img2).to(cfg.DEVICE)
        
        with torch.no_grad():
            distance = loss_fn(tensor1, tensor2)
        
        return float(distance.cpu().numpy())
    except Exception as e:
        print(f"LPIPS计算错误: {e}")
        return float('nan')


def load_test_dataset_for_evaluation(run_dir: str) -> Tuple[Optional[List[np.ndarray]], Optional[List[int]]]:
    """加载测试数据集用于定量评估
    
    Args:
        run_dir: 运行目录
        
    Returns:
        (test_images, test_labels) 或 (None, None) 如果无法加载
    """
    try:
        # 尝试从配置文件获取数据集信息
        config_path = os.path.join(run_dir, 'config.py')
        if os.path.exists(config_path):
            # 动态导入配置
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_config", config_path)
            temp_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_config)
            
            dataset_name = getattr(temp_config, 'DATASET_NAME', None)
            data_root = getattr(temp_config, 'DATA_ROOT', './data')
            print(f"检测到数据集: {dataset_name}, 数据目录: {data_root}")
            
            # 导入必要的模块
            import torchvision.transforms as transforms
            from torchvision import datasets
            from torch.utils.data import DataLoader, Subset
            import medmnist
            
            # 构建数据转换（与训练时相同）
            if hasattr(temp_config, 'IN_CHANNELS') and temp_config.IN_CHANNELS == 1:
                norm_mean, norm_std = (0.5,), (0.5,)
            else:
                norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            
            # 构建转换列表
            transform_list = [transforms.Resize(getattr(temp_config, 'TARGET_SIZE', (32, 32)))]
            
            # 对于 SHAPES_CLASSIFICATION 数据集，添加灰度转换
            if dataset_name == 'SHAPES_CLASSIFICATION':
                transform_list.append(transforms.Grayscale(num_output_channels=1))
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
            data_transform = transforms.Compose(transform_list)
            
            test_dataset = None
            test_labels = []
            
            # 根据数据集名称加载测试数据
            if dataset_name == 'FASHION_MNIST':
                test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=False, transform=data_transform)
                test_labels = test_dataset.targets.tolist()
                
            elif dataset_name == 'MNIST':
                test_dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=data_transform)
                test_labels = test_dataset.targets.tolist()
                
            elif dataset_name == 'GTSRB':
                # 处理 GTSRB 子集
                gtsrb_subset_indices = getattr(temp_config, 'GTSRB_SUBSET_INDICES', None)
                target_transform = None
                
                if gtsrb_subset_indices is not None:
                    # 创建标签映射: 原始ID -> 0..N-1
                    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(gtsrb_subset_indices)}
                    target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))
                    
                    test_dataset = datasets.GTSRB(root=data_root, split='test', download=False,
                                                  transform=data_transform, target_transform=target_transform)
                    
                    # 过滤数据集
                    subset_set = set(gtsrb_subset_indices)
                    test_indices = [i for i, (_, label) in enumerate(test_dataset._samples) if label in subset_set]
                    test_dataset = Subset(test_dataset, test_indices)
                    
                    # 获取过滤后的标签
                    test_labels = [mapping[test_dataset.dataset._samples[i][1]] for i in test_indices]
                else:
                    test_dataset = datasets.GTSRB(root=data_root, split='test', download=False, transform=data_transform)
                    test_labels = [label for _, label in test_dataset._samples]
                    
            elif dataset_name == 'BLOOD_MNIST':
                test_dataset = medmnist.BloodMNIST(split='test', transform=data_transform, download=False, root=data_root)
                test_labels = test_dataset.labels.squeeze().tolist()
                
            elif dataset_name == 'SHAPES_CLASSIFICATION':
                # 对于自定义形状数据集，使用 ImageFolder
                from torchvision.datasets import ImageFolder
                
                # 尝试多个可能的路径（优先检查包含实际图像的路径）
                possible_paths = [
                    os.path.join(data_root, 'Shapes_Classification', 'archive(6)', 'shapes'),  # 实际图像路径
                    os.path.join(data_root, 'Shapes_Classification', 'shapes'),
                    os.path.join(data_root, 'shapes_classification'),
                    os.path.join(data_root, 'Shapes_Classification'),  # 最后检查根目录
                ]
                
                shapes_data_dir = None
                for path in possible_paths:
                    if os.path.exists(path):
                        # 检查路径是否包含图像文件
                        import glob
                        image_files = glob.glob(os.path.join(path, '*', '*.jpg')) + \
                                     glob.glob(os.path.join(path, '*', '*.png')) + \
                                     glob.glob(os.path.join(path, '*', '*.jpeg'))
                        if image_files or path == os.path.join(data_root, 'Shapes_Classification', 'archive(6)', 'shapes'):
                            shapes_data_dir = path
                            print(f"找到形状数据集目录: {shapes_data_dir}")
                            break
                        else:
                            print(f"路径 {path} 存在但不包含图像文件，跳过")
                
                if shapes_data_dir is not None:
                    try:
                        # 加载整个数据集，然后分割（与训练时相同）
                        full_dataset = ImageFolder(root=shapes_data_dir, transform=data_transform)
                        # 使用固定的随机种子进行分割，确保与训练时相同的测试集
                        torch.manual_seed(42)
                        indices = torch.randperm(len(full_dataset))
                        # 假设训练集占80%，测试集占20%
                        test_size = int(0.2 * len(full_dataset))
                        test_indices = indices[:test_size]
                        test_dataset = Subset(full_dataset, test_indices)
                        test_labels = [full_dataset.targets[i] for i in test_indices]
                    except Exception as e:
                        print(f"加载形状数据集时出错: {e}")
                        return None, None
                else:
                    print(f"警告: 未找到形状数据集目录，尝试的路径: {possible_paths}")
                    return None, None
                    
            elif dataset_name == 'CIFAR10':
                test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=data_transform)
                test_labels = test_dataset.targets
                
            elif dataset_name == 'CIFAR100':
                test_dataset = datasets.CIFAR100(root=data_root, train=False, download=False, transform=data_transform)
                test_labels = test_dataset.targets
                
            else:
                print(f"不支持的数据集: {dataset_name}")
                return None, None
            
            if test_dataset is not None:
                print(f"成功加载测试数据集: {dataset_name}, 测试样本数: {len(test_dataset)}")
                
                # 提取测试图像（转换为numpy数组）
                test_images = []
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                for batch_data, _ in test_loader:
                    # 将张量转换为numpy数组并调整维度顺序
                    batch_np = batch_data.numpy()
                    # 从 [B, C, H, W] 转换为 [B, H, W, C] 用于图像处理
                    if batch_np.shape[1] == 1:  # 灰度图像
                        batch_np = batch_np.transpose(0, 2, 3, 1).squeeze(-1)
                    else:  # RGB图像
                        batch_np = batch_np.transpose(0, 2, 3, 1)
                    
                    # 反归一化到 [0, 1] 范围
                    batch_np = (batch_np * norm_std[0]) + norm_mean[0]
                    batch_np = np.clip(batch_np, 0, 1)
                    
                    for img in batch_np:
                        test_images.append(img)
                
                # 如果提取的图像数量与标签数量不一致，只取前N个
                if len(test_images) != len(test_labels):
                    min_len = min(len(test_images), len(test_labels))
                    test_images = test_images[:min_len]
                    test_labels = test_labels[:min_len]
                    print(f"调整数据长度: 图像={len(test_images)}, 标签={len(test_labels)}")
                
                return test_images, test_labels
            
    except Exception as e:
        print(f"加载测试数据集时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 尝试从数据目录加载
    data_dir = os.path.join(os.path.dirname(run_dir), '..', 'data')
    if os.path.exists(data_dir):
        print(f"检测到数据目录: {data_dir}")
        # 这里可以添加具体的数据加载逻辑
        # 暂时返回None
    
    print("无法自动加载测试数据集，请手动提供测试数据或跳过定量评估")
    return None, None


def evaluate_visualization_quality(run_dir: str, decoded_images: np.ndarray,
                                   test_images: List[np.ndarray] = None,
                                   test_labels: List[int] = None) -> Dict[str, Any]:
    """定量评估可视化质量
    
    Args:
        run_dir: 运行目录
        decoded_images: 解码后的规则可视化图像，形状为 [num_rules, C, H, W]
        test_images: 可选的测试图像列表
        test_labels: 可选的测试图像标签
        
    Returns:
        评估结果字典
    """
    print(f"\n>>> 开始定量评估可视化质量")
    
    if decoded_images is None or len(decoded_images) == 0:
        print("警告: 没有解码图像，跳过定量评估")
        return {}
    
    num_rules = len(decoded_images)
    print(f"评估 {num_rules} 条规则的可视化质量...")
    
    # 准备评估结果
    results = {
        'rule_index': list(range(num_rules)),
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    # 如果没有测试图像，使用规则图像自身进行比较（自相似性）
    if test_images is None:
        print("未提供测试图像，计算规则图像的自相似性指标...")
        # 这里可以计算规则图像之间的相似性
        # 暂时跳过详细实现
        return results
    
    # 如果有测试图像，计算与测试图像的相似性
    print(f"计算与 {len(test_images)} 张测试图像的相似性...")
    
    # 为每条规则计算与第一张测试图像的指标
    if len(test_images) > 0 and num_rules > 0:
        test_img = test_images[0]
        
        # 准备测试图像显示格式
        if len(test_img.shape) == 3 and test_img.shape[0] in [1, 3]:  # [C, H, W]
            test_img_display = np.transpose(test_img, (1, 2, 0))
            if test_img_display.shape[2] == 1:
                test_img_display = test_img_display.squeeze()
        else:
            test_img_display = test_img
        
        # 为每条规则计算指标
        for i in range(num_rules):
            rule_img = decoded_images[i]
            
            # 调整规则图像格式
            if len(rule_img.shape) == 3 and rule_img.shape[0] in [1, 3]:  # [C, H, W]
                rule_img_display = np.transpose(rule_img, (1, 2, 0))
                if rule_img_display.shape[2] == 1:
                    rule_img_display = rule_img_display.squeeze()
            else:
                rule_img_display = rule_img
            
            # 确保测试图像与规则图像尺寸相同
            # 将测试图像调整到规则图像的尺寸
            if test_img_display.shape != rule_img_display.shape:
                from skimage.transform import resize
                try:
                    # 调整测试图像到规则图像的尺寸
                    test_img_resized = resize(test_img_display, rule_img_display.shape,
                                             order=1, preserve_range=True, anti_aliasing=True)
                    test_img_display = test_img_resized
                except Exception as e:
                    print(f"调整图像尺寸时出错: {e}")
                    # 如果调整失败，使用原始图像
            
            # 计算指标
            psnr_value = calculate_psnr(test_img_display, rule_img_display)
            ssim_value = calculate_ssim(test_img_display, rule_img_display)
            lpips_value = calculate_lpips(test_img_display, rule_img_display)
            
            # 保存到结果
            results['psnr'].append(psnr_value)
            results['ssim'].append(ssim_value)
            results['lpips'].append(lpips_value)
            
            if i == 0:
                print(f"规则 0 示例指标 - PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.3f}, LPIPS: {lpips_value:.3f}")
        
        print(f"已为 {num_rules} 条规则计算了指标")
    
    # 生成评估报告
    generate_quantitative_report(run_dir, results)
    
    return results


def generate_quantitative_report(run_dir: str, results: Dict[str, Any]):
    """生成定量评估报告"""
    if not results or not results.get('psnr'):
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存CSV文件
    csv_path = os.path.join(run_dir, 'visualization_quality_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"可视化质量指标已保存至: {csv_path}")
    
    # 计算汇总统计
    summary = {}
    metrics = ['psnr', 'ssim', 'lpips']
    
    for metric in metrics:
        if metric in df.columns and len(df[metric]) > 0:
            values = df[metric].dropna()
            if len(values) > 0:
                summary[f'{metric}_mean'] = float(values.mean())
                summary[f'{metric}_std'] = float(values.std())
                summary[f'{metric}_min'] = float(values.min())
                summary[f'{metric}_max'] = float(values.max())
    
    # 保存汇总统计
    summary_df = pd.DataFrame([summary])
    summary_csv_path = os.path.join(run_dir, 'visualization_quality_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"可视化质量汇总统计已保存至: {summary_csv_path}")
    
    # 生成可视化图表
    generate_quality_visualization(run_dir, df, summary)
    
    # 生成文本报告
    generate_text_report(run_dir, summary)


def generate_quality_visualization(run_dir: str, df: pd.DataFrame, summary: Dict[str, float]):
    """生成质量可视化图表"""
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PSNR分布
    if 'psnr' in df.columns and not df['psnr'].isna().all():
        ax = axes[0]
        psnr_values = df['psnr'].dropna()
        ax.hist(psnr_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(psnr_values.mean(), color='red', linestyle='--',
                  label=f'均值: {psnr_values.mean():.2f} dB')
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('频数')
        ax.set_title('PSNR分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # SSIM分布
    if 'ssim' in df.columns and not df['ssim'].isna().all():
        ax = axes[1]
        ssim_values = df['ssim'].dropna()
        ax.hist(ssim_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(ssim_values.mean(), color='red', linestyle='--',
                  label=f'均值: {ssim_values.mean():.3f}')
        ax.set_xlabel('SSIM')
        ax.set_ylabel('频数')
        ax.set_title('SSIM分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # LPIPS分布
    if 'lpips' in df.columns and not df['lpips'].isna().all():
        ax = axes[2]
        lpips_values = df['lpips'].dropna()
        ax.hist(lpips_values, bins=15, alpha=0.7, color='salmon', edgecolor='black')
        ax.axvline(lpips_values.mean(), color='red', linestyle='--',
                  label=f'均值: {lpips_values.mean():.3f}')
        ax.set_xlabel('LPIPS')
        ax.set_ylabel('频数')
        ax.set_title('LPIPS分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(run_dir, 'visualization_quality_summary.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化质量图表已保存至: {chart_path}")


def generate_text_report(run_dir: str, summary: Dict[str, float]):
    """生成文本评估报告"""
    report_path = os.path.join(run_dir, 'quantitative_evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("       可视化质量定量评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("评估时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("\n")
        
        f.write("一、评估指标说明:\n")
        f.write("1. PSNR (峰值信噪比): 衡量图像重建质量，值越大越好，单位dB\n")
        f.write("2. SSIM (结构相似性指数): 衡量图像结构相似性，范围[-1, 1]，值越大越好\n")
        f.write("3. LPIPS (学习感知图像块相似度): 基于深度学习的感知相似度，值越小越好\n")
        f.write("\n")
        
        f.write("二、汇总统计结果:\n")
        f.write("-" * 40 + "\n")
        
        metrics_info = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'lpips': 'LPIPS'
        }
        
        for metric_base, metric_name in metrics_info.items():
            mean_key = f'{metric_base}_mean'
            std_key = f'{metric_base}_std'
            min_key = f'{metric_base}_min'
            max_key = f'{metric_base}_max'
            
            if mean_key in summary:
                f.write(f"{metric_name}:\n")
                f.write(f"  平均值: {summary[mean_key]:.3f}\n")
                if std_key in summary:
                    f.write(f"  标准差: {summary[std_key]:.3f}\n")
                if min_key in summary:
                    f.write(f"  最小值: {summary[min_key]:.3f}\n")
                if max_key in summary:
                    f.write(f"  最大值: {summary[max_key]:.3f}\n")
                f.write("\n")
        
        f.write("三、评估结论:\n")
        f.write("-" * 40 + "\n")
        
        # 简单的质量评估
        if 'psnr_mean' in summary:
            psnr_mean = summary['psnr_mean']
            if psnr_mean > 30:
                f.write("PSNR指标优秀 (>30 dB)，图像重建质量很高。\n")
            elif psnr_mean > 20:
                f.write("PSNR指标良好 (20-30 dB)，图像重建质量可接受。\n")
            else:
                f.write("PSNR指标一般 (<20 dB)，图像重建质量有待提高。\n")
        
        if 'ssim_mean' in summary:
            ssim_mean = summary['ssim_mean']
            if ssim_mean > 0.8:
                f.write("SSIM指标优秀 (>0.8)，图像结构保持很好。\n")
            elif ssim_mean > 0.6:
                f.write("SSIM指标良好 (0.6-0.8)，图像结构保持较好。\n")
            else:
                f.write("SSIM指标一般 (<0.6)，图像结构保持有待提高。\n")
        
        if 'lpips_mean' in summary:
            lpips_mean = summary['lpips_mean']
            if lpips_mean < 0.2:
                f.write("LPIPS指标优秀 (<0.2)，感知相似度很高。\n")
            elif lpips_mean < 0.4:
                f.write("LPIPS指标良好 (0.2-0.4)，感知相似度可接受。\n")
            else:
                f.write("LPIPS指标一般 (>0.4)，感知相似度有待提高。\n")
        
        f.write("\n四、建议:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 如果PSNR较低，考虑优化解码器训练过程\n")
        f.write("2. 如果SSIM较低，考虑改进特征提取和规则表示\n")
        f.write("3. 如果LPIPS较高，考虑使用更先进的感知损失函数\n")
        f.write("4. 可以尝试不同的解码器架构进行比较\n")
    
    print(f"定量评估报告已保存至: {report_path}")


def run_visualization(run_dir, compare_decoders=True):
    """[修改] 接收 run_dir 参数，支持对比可视化
    
    Args:
        run_dir: 运行目录
        compare_decoders: 是否进行两种解码器的对比可视化
    """
    print(f"\n>>> 开始规则可视化 (Rule Visualization): {run_dir}")
    print(f">>> 对比模式: {'开启' if compare_decoders else '关闭'}")

    # 加载模型
    model_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE,weights_only=False)
    configure_model_from_checkpoint(checkpoint)

    # 加载模型
    model = FullModel().to(cfg.DEVICE)
    # 使用 strict=False 允许加载形状不完全匹配的检查点
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    if compare_decoders:
        # 加载两种解码器进行对比
        decoders = get_decoder_from_config(run_dir, decoder_type='both')
        
        if decoders and len(decoders) >= 2:
            print(f"\n已加载 {len(decoders)} 种解码器: {list(decoders.keys())}")
            
            # 进行对比可视化
            visualize_comparison_rules(model, decoders, run_dir)
            
            # 分别对每种解码器进行可视化
            for decoder_type, decoder in decoders.items():
                visualize_single_scale_rules(model, decoder, run_dir, decoder_type=decoder_type)
        else:
            print("警告: 解码器数量不足，进行单解码器可视化")
            decoder = get_decoder_from_config(run_dir, decoder_type='standard')
            if decoder:
                visualize_single_scale_rules(model, decoder, run_dir, decoder_type='standard')
    else:
        # 原有逻辑：加载单个解码器
        decoder = get_decoder_from_config(run_dir, decoder_type='standard')
        if decoder:
            visualize_single_scale_rules(model, decoder, run_dir, decoder_type='standard')

    # 可视化注意力权重热图
    if num_rules > 0:
        visualize_attention_heatmap(classifier, num_rules, run_dir)

    # 创建规则摘要表格
    if num_rules > 0:
        consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
        consequents_prob = scipy.special.softmax(consequents, axis=1)
        predicted_classes = np.argmax(consequents_prob, axis=1)
        confidences = np.max(consequents_prob, axis=1)

        rule_specific_attention = None
        if cfg.USE_ATTENTION and classifier.alpha is not None:
            alpha = classifier.alpha[:num_rules].detach()
            rule_specific_attention = F.softmax(alpha, dim=1)

        create_rule_summary_table(run_dir, predicted_classes, confidences, rule_specific_attention)

    # 定量评估可视化质量
    if num_rules > 0:
        print(f"\n>>> 开始定量评估可视化质量")
        
        # 尝试加载测试数据集进行定量评估
        test_images, test_labels = load_test_dataset_for_evaluation(run_dir)
        
        if test_images is not None and len(test_images) > 0:
            print(f"加载了 {len(test_images)} 张测试图像进行定量评估")
            
            # 获取解码后的规则图像
            # 这里需要从可视化过程中获取解码图像
            # 由于可视化函数已经生成了解码图像，我们需要修改代码来保存它们
            # 暂时使用简化版本：在单解码器模式下进行评估
            
            # 无论是单解码器还是对比解码器模式，都进行定量评估
            # 首先尝试获取标准解码器进行评估
            decoder = get_decoder_from_config(run_dir, decoder_type='standard')
            if decoder:
                # 重新解码规则中心以获取图像
                centers = classifier.centers[:num_rules].detach()
                centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)
                
                # 获取注意力权重
                rule_specific_attention = None
                if cfg.USE_ATTENTION and classifier.alpha is not None:
                    alpha = classifier.alpha[:num_rules].detach()
                    rule_specific_attention = F.softmax(alpha, dim=1)
                
                # 解码规则中心
                with torch.no_grad():
                    decoded_images = []
                    for i in range(num_rules):
                        rule_center = centers_reshaped[i:i+1]
                        if rule_specific_attention is not None:
                            rule_attention = rule_specific_attention[i:i+1]
                            if isinstance(decoder, (AttentionGuidedDecoder, AttentionGuidedMultiScaleDecoder)):
                                decoded = decoder(rule_center, rule_attention)
                            else:
                                decoded = decoder(rule_center)
                        else:
                            decoded = decoder(rule_center)
                        
                        # 反归一化
                        decoded = decoded * 0.5 + 0.5
                        decoded = torch.clamp(decoded, 0, 1)
                        decoded_images.append(decoded.cpu().numpy())
                
                if decoded_images:
                    decoded_images_array = np.array(decoded_images)
                    
                    # 调用定量评估函数
                    evaluate_visualization_quality(
                        run_dir=run_dir,
                        decoded_images=decoded_images_array,
                        test_images=test_images,
                        test_labels=test_labels
                    )
                    
                    # 如果是对比解码器模式，也评估其他解码器
                    if compare_decoders:
                        print("\n>>> 开始对比解码器的定量评估")
                        # 获取所有解码器
                        decoders = get_decoder_from_config(run_dir, decoder_type='both')
                        if decoders and len(decoders) > 1:
                            for decoder_name, decoder_obj in decoders.items():
                                if decoder_name != 'standard':  # 标准解码器已经评估过了
                                    print(f"\n评估 {decoder_name} 解码器...")
                                    with torch.no_grad():
                                        decoded_images_other = []
                                        for i in range(num_rules):
                                            rule_center = centers_reshaped[i:i+1]
                                            if rule_specific_attention is not None:
                                                rule_attention = rule_specific_attention[i:i+1]
                                                if isinstance(decoder_obj, (AttentionGuidedDecoder, AttentionGuidedMultiScaleDecoder)):
                                                    decoded = decoder_obj(rule_center, rule_attention)
                                                else:
                                                    decoded = decoder_obj(rule_center)
                                            else:
                                                decoded = decoder_obj(rule_center)
                                            
                                            # 反归一化
                                            decoded = decoded * 0.5 + 0.5
                                            decoded = torch.clamp(decoded, 0, 1)
                                            decoded_images_other.append(decoded.cpu().numpy())
                                    
                                    if decoded_images_other:
                                        decoded_images_other_array = np.array(decoded_images_other)
                                        
                                        # 为其他解码器创建单独的评估目录
                                        decoder_eval_dir = os.path.join(run_dir, f'evaluation_{decoder_name}')
                                        os.makedirs(decoder_eval_dir, exist_ok=True)
                                        
                                        # 调用定量评估函数
                                        evaluate_visualization_quality(
                                            run_dir=decoder_eval_dir,
                                            decoded_images=decoded_images_other_array,
                                            test_images=test_images,
                                            test_labels=test_labels
                                        )
            else:
                print("未找到解码器，跳过定量评估")
        else:
            print("未找到测试数据集，跳过定量评估")
    
    print(f"\n规则可视化完成！所有结果已保存到: {run_dir}")


# ---------------------------------------------------------------------------
# Quantitative rule-visualization evaluation, v2.
# These definitions intentionally override the older simplified evaluation
# above. The older code compared rule images with only one test image; this
# version evaluates every test sample against its matched rule image.
# ---------------------------------------------------------------------------

def _eval_to_display_image(img: np.ndarray) -> np.ndarray:
    """Convert CHW/HWC/gray image arrays to HW or HWC float arrays in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def _eval_to_three_channel_hwc(img: np.ndarray) -> np.ndarray:
    img = _eval_to_display_image(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def _eval_resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape == ref.shape:
        return img.astype(np.float32)
    from skimage.transform import resize
    return resize(img, ref.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def _eval_prepare_lpips_tensor(img: np.ndarray) -> torch.Tensor:
    from skimage.transform import resize

    img = _eval_to_three_channel_hwc(img)
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        scale = max(64 / h, 64 / w)
        img = resize(
            img,
            (int(math.ceil(h * scale)), int(math.ceil(w * scale)), 3),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) * 2.0 - 1.0


def calculate_lpips(img1: np.ndarray, img2: np.ndarray, loss_fn=None) -> float:
    """Calculate LPIPS while allowing a caller-supplied reusable model."""
    if not HAS_LPIPS:
        return float('nan')
    try:
        img1 = _eval_to_three_channel_hwc(img1)
        img2 = _eval_resize_like(_eval_to_three_channel_hwc(img2), img1)

        if loss_fn is None:
            loss_fn = lpips.LPIPS(net='alex', verbose=False).to(cfg.DEVICE)
            loss_fn.eval()

        tensor1 = _eval_prepare_lpips_tensor(img1).to(cfg.DEVICE)
        tensor2 = _eval_prepare_lpips_tensor(img2).to(cfg.DEVICE)
        with torch.no_grad():
            distance = loss_fn(tensor1, tensor2)
        return float(distance.detach().cpu().item())
    except Exception as e:
        print(f"LPIPS计算错误: {e}")
        return float('nan')


def _denormalize_batch(batch: torch.Tensor, norm_mean: Tuple[float, ...],
                       norm_std: Tuple[float, ...]) -> List[np.ndarray]:
    mean = torch.tensor(norm_mean, dtype=batch.dtype, device=batch.device).view(1, -1, 1, 1)
    std = torch.tensor(norm_std, dtype=batch.dtype, device=batch.device).view(1, -1, 1, 1)
    batch = torch.clamp(batch * std + mean, 0.0, 1.0).detach().cpu().numpy()
    return [_eval_to_display_image(img) for img in batch]


def _images_to_fid_tensor(images: List[np.ndarray], as_uint8: bool = False) -> torch.Tensor:
    tensors = []
    for img in images:
        img3 = _eval_to_three_channel_hwc(img)
        tensors.append(torch.from_numpy(img3).permute(2, 0, 1).float())
    batch = torch.stack(tensors, dim=0)
    if as_uint8:
        return (batch * 255.0).round().clamp(0, 255).to(torch.uint8)
    return batch.clamp(0.0, 1.0)


def _make_fid_metric():
    if not HAS_FID:
        return None, False
    try:
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(cfg.DEVICE)
        return metric, False
    except TypeError:
        try:
            metric = FrechetInceptionDistance(feature=2048).to(cfg.DEVICE)
            return metric, True
        except Exception as e:
            print(f"FID初始化失败: {e}")
            return None, False
    except Exception as e:
        print(f"FID初始化失败: {e}")
        return None, False


def _load_evaluation_loader(run_dir: str, batch_size: int = 64) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Load a normalized test DataLoader using the run's saved config."""
    try:
        import importlib.util
        import torchvision.transforms as transforms
        from torchvision import datasets
        from torch.utils.data import DataLoader, Subset
        import medmnist

        config_path = os.path.join(run_dir, 'config.py')
        if not os.path.exists(config_path):
            print(f"未找到运行配置文件: {config_path}")
            return None, None

        spec = importlib.util.spec_from_file_location("temp_eval_config", config_path)
        temp_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_config)

        dataset_name = getattr(temp_config, 'DATASET_NAME', cfg.DATASET_NAME)
        data_root = getattr(temp_config, 'DATA_ROOT', cfg.DATA_ROOT)
        target_size = getattr(temp_config, 'TARGET_SIZE', cfg.TARGET_SIZE)
        in_channels = getattr(temp_config, 'IN_CHANNELS', cfg.IN_CHANNELS)
        seed = getattr(temp_config, 'SEED', 42)

        norm_mean = (0.5,) if in_channels == 1 else (0.5, 0.5, 0.5)
        norm_std = (0.5,) if in_channels == 1 else (0.5, 0.5, 0.5)
        transform_list = [transforms.Resize(target_size)]
        if dataset_name in ['GEOMETRIC_SHAPES', 'SHAPES_CLASSIFICATION']:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
        data_transform = transforms.Compose(transform_list)

        if dataset_name == 'FASHION_MNIST':
            test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=False, transform=data_transform)
        elif dataset_name == 'MNIST':
            test_dataset = datasets.MNIST(root=data_root, train=False, download=False, transform=data_transform)
        elif dataset_name == 'BLOOD_MNIST':
            test_dataset = medmnist.BloodMNIST(split='test', transform=data_transform, download=False, root=data_root)
        elif dataset_name == 'GTSRB':
            subset_indices = getattr(temp_config, 'GTSRB_SUBSET_INDICES', None)
            target_transform = None
            if subset_indices is not None:
                mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subset_indices)}
                target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))
            test_dataset = datasets.GTSRB(
                root=data_root,
                split='test',
                download=False,
                transform=data_transform,
                target_transform=target_transform
            )
            if subset_indices is not None:
                subset_set = set(subset_indices)
                test_indices = [i for i, (_, label) in enumerate(test_dataset._samples) if label in subset_set]
                test_dataset = Subset(test_dataset, test_indices)
        elif dataset_name == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=data_transform)
        elif dataset_name == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root=data_root, train=False, download=False, transform=data_transform)
        elif dataset_name == 'GEOMETRIC_SHAPES':
            full_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'geometric_shapes'), transform=data_transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
            )
        elif dataset_name == 'MIO_TCD_CLASSIFICATION':
            full_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'MIO-TCD-Classification'), transform=data_transform)
            train_size = int((5 / 6) * len(full_dataset))
            test_size = len(full_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
            )
        elif dataset_name == 'VEHICLES':
            full_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'Vehicles'), transform=data_transform)
            train_size = int((4 / 5) * len(full_dataset))
            test_size = len(full_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
            )
        elif dataset_name == 'SHAPES_CLASSIFICATION':
            dataset_path = os.path.join(data_root, 'Shapes_Classification', 'archive(6)', 'shapes')
            full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
            train_size = int((4 / 5) * len(full_dataset))
            test_size = len(full_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
            )
        else:
            print(f"不支持的评估数据集: {dataset_name}")
            return None, None

        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        metadata = {
            'dataset_name': dataset_name,
            'num_samples': len(test_dataset),
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }
        print(f"成功加载评估数据集: {dataset_name}, 样本数: {len(test_dataset)}")
        return loader, metadata
    except Exception as e:
        print(f"加载评估数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _prepare_decoded_rule_images(model, decoder) -> Optional[np.ndarray]:
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    if num_rules <= 0:
        return None

    centers = classifier.centers[:num_rules].detach()
    centers_reshaped = centers.view(num_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT)

    rule_specific_attention = None
    if cfg.USE_ATTENTION and classifier.alpha is not None:
        alpha = classifier.alpha[:num_rules].detach()
        rule_specific_attention = F.softmax(alpha, dim=1)

    decoded = decode_rule_centers(decoder, centers_reshaped, rule_specific_attention)
    if decoded.ndim == 5 and decoded.shape[1] == 1:
        decoded = decoded[:, 0]
    return decoded


def _evaluate_rule_visualization_dataset(run_dir: str, model, decoded_images: np.ndarray,
                                         eval_loader, eval_meta: Dict[str, Any],
                                         decoder_name: str = 'standard') -> Dict[str, Any]:
    print(f"\n>>> 开始定量评估: {decoder_name}")
    if decoded_images is None or len(decoded_images) == 0:
        print("没有可评估的规则可视化图片，跳过。")
        return {}

    model.eval()
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()
    consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
    rule_probs = scipy.special.softmax(consequents, axis=1)
    rule_classes = np.argmax(rule_probs, axis=1)

    decoded_display = [_eval_to_display_image(img) for img in decoded_images]
    class_to_rules = defaultdict(list)
    for rule_idx, rule_class in enumerate(rule_classes):
        class_to_rules[int(rule_class)].append(rule_idx)

    lpips_loss = None
    if HAS_LPIPS:
        try:
            lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(cfg.DEVICE)
            lpips_loss.eval()
        except Exception as e:
            print(f"LPIPS初始化失败，将写入NaN: {e}")

    fid_metric, fid_requires_uint8 = _make_fid_metric()
    if not HAS_SKIMAGE:
        print("缺少 scikit-image，PSNR/SSIM 将写入 NaN。")
    if not HAS_FID:
        print("缺少 torchmetrics FID，FID 将写入 NaN。")

    rows = []
    sample_index = 0
    norm_mean = eval_meta['norm_mean']
    norm_std = eval_meta['norm_std']

    for batch_idx, batch in enumerate(eval_loader):
        data, targets = batch[0], batch[1]
        if targets.ndim > 1:
            targets = targets.squeeze()
        labels = targets.detach().cpu().numpy().astype(int).reshape(-1)

        data = data.to(cfg.DEVICE)
        with torch.no_grad():
            features = model.extractor(data)
            activations = classifier.get_rule_activations(features)
        if activations is None:
            print("规则激活为空，停止评估。")
            break
        activations_np = activations.detach().cpu().numpy()
        real_images = _denormalize_batch(data, norm_mean, norm_std)

        fake_images_for_fid = []
        for i, real_img in enumerate(real_images):
            label = int(labels[i])
            candidate_rules = class_to_rules.get(label, [])
            class_rule_found = len(candidate_rules) > 0

            if class_rule_found:
                candidate_acts = activations_np[i, candidate_rules]
                matched_rule = int(candidate_rules[int(np.argmax(candidate_acts))])
            else:
                matched_rule = int(np.argmax(activations_np[i]))

            rule_img = decoded_display[matched_rule]
            rule_img_for_metrics = _eval_resize_like(rule_img, real_img)

            psnr_value = calculate_psnr(real_img, rule_img_for_metrics)
            ssim_value = calculate_ssim(real_img, rule_img_for_metrics)
            lpips_value = calculate_lpips(real_img, rule_img_for_metrics, lpips_loss) if lpips_loss is not None else float('nan')

            rows.append({
                'sample_index': sample_index,
                'label': label,
                'matched_rule_index': matched_rule,
                'matched_rule_class': int(rule_classes[matched_rule]),
                'rule_activation': float(activations_np[i, matched_rule]),
                'class_rule_found': bool(class_rule_found),
                'psnr': psnr_value,
                'ssim': ssim_value,
                'lpips': lpips_value,
            })

            fake_images_for_fid.append(rule_img_for_metrics)
            sample_index += 1

        if fid_metric is not None and real_images and fake_images_for_fid:
            try:
                real_tensor = _images_to_fid_tensor(real_images, as_uint8=fid_requires_uint8).to(cfg.DEVICE)
                fake_tensor = _images_to_fid_tensor(fake_images_for_fid, as_uint8=fid_requires_uint8).to(cfg.DEVICE)
                fid_metric.update(real_tensor, real=True)
                fid_metric.update(fake_tensor, real=False)
            except Exception as e:
                print(f"FID批次更新失败，将跳过FID: {e}")
                fid_metric = None

        if (batch_idx + 1) % 10 == 0:
            print(f"已评估 {sample_index} / {eval_meta['num_samples']} 张图片...")

    fid_value = float('nan')
    if fid_metric is not None:
        try:
            fid_value = float(fid_metric.compute().detach().cpu().item())
        except Exception as e:
            print(f"FID计算失败: {e}")

    results = {
        'rows': rows,
        'fid': fid_value,
        'dataset_name': eval_meta.get('dataset_name', ''),
        'decoder_name': decoder_name,
        'num_samples': len(rows),
    }
    generate_quantitative_report(run_dir, results)
    return results


def generate_quantitative_report(run_dir: str, results: Dict[str, Any]):
    if not results or not results.get('rows'):
        print("没有定量评估结果可保存。")
        return

    os.makedirs(run_dir, exist_ok=True)
    df = pd.DataFrame(results['rows'])
    csv_path = os.path.join(run_dir, 'visualization_quality_metrics.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"逐样本指标已保存至: {csv_path}")

    summary = {
        'dataset_name': results.get('dataset_name', ''),
        'decoder_name': results.get('decoder_name', ''),
        'num_samples': int(results.get('num_samples', len(df))),
        'fid': float(results.get('fid', float('nan'))),
        'class_rule_found_rate': float(df['class_rule_found'].mean()) if 'class_rule_found' in df else float('nan'),
    }
    for metric in ['psnr', 'ssim', 'lpips']:
        if metric in df.columns:
            values = df[metric].dropna()
            summary[f'{metric}_mean'] = float(values.mean()) if len(values) > 0 else float('nan')
            summary[f'{metric}_std'] = float(values.std()) if len(values) > 0 else float('nan')
            summary[f'{metric}_min'] = float(values.min()) if len(values) > 0 else float('nan')
            summary[f'{metric}_max'] = float(values.max()) if len(values) > 0 else float('nan')

    summary_csv_path = os.path.join(run_dir, 'visualization_quality_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"汇总指标已保存至: {summary_csv_path}")

    generate_quality_visualization(run_dir, df, summary)
    generate_text_report(run_dir, summary)


def generate_quality_visualization(run_dir: str, df: pd.DataFrame, summary: Dict[str, float]):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_styles = [
        ('psnr', 'PSNR (dB)', 'PSNR Distribution', 'skyblue', '{:.2f}'),
        ('ssim', 'SSIM', 'SSIM Distribution', 'lightgreen', '{:.3f}'),
        ('lpips', 'LPIPS', 'LPIPS Distribution', 'salmon', '{:.3f}'),
    ]

    for ax, (metric, xlabel, title, color, fmt) in zip(axes, metric_styles):
        if metric in df.columns and not df[metric].isna().all():
            values = df[metric].dropna()
            ax.hist(values, bins=15, alpha=0.75, color=color, edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--',
                       label=f"Mean: {fmt.format(values.mean())}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fid_value = summary.get('fid', float('nan'))
    fid_text = 'NaN' if pd.isna(fid_value) else f'{fid_value:.3f}'
    fig.suptitle(f"Rule Visualization Quality | FID: {fid_text}", fontsize=13)
    plt.tight_layout()
    chart_path = os.path.join(run_dir, 'visualization_quality_summary.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"指标分布图已保存至: {chart_path}")


def generate_text_report(run_dir: str, summary: Dict[str, float]):
    report_path = os.path.join(run_dir, 'quantitative_evaluation_report.txt')
    fid_value = summary.get('fid', float('nan'))
    fid_text = 'NaN' if pd.isna(fid_value) else f'{fid_value:.4f}'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 64 + "\n")
        f.write("规则可视化质量定量评估报告\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集: {summary.get('dataset_name', '')}\n")
        f.write(f"Decoder: {summary.get('decoder_name', '')}\n")
        f.write(f"样本数: {summary.get('num_samples', 0)}\n")
        f.write("匹配策略: 优先选择真实标签对应规则中激活最高的规则；若不存在同类规则，则回退到全局最高激活规则。\n")
        f.write("FID口径: 将所有原图作为real集合，所有匹配规则图作为fake集合，全数据集计算一次。\n\n")

        f.write("一、汇总指标\n")
        f.write("-" * 40 + "\n")
        metric_names = {'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'lpips': 'LPIPS'}
        for key, name in metric_names.items():
            mean = summary.get(f'{key}_mean', float('nan'))
            std = summary.get(f'{key}_std', float('nan'))
            min_v = summary.get(f'{key}_min', float('nan'))
            max_v = summary.get(f'{key}_max', float('nan'))
            f.write(f"{name}: mean={mean:.4f}, std={std:.4f}, min={min_v:.4f}, max={max_v:.4f}\n")
        f.write(f"FID: {fid_text}\n")
        f.write(f"真实类别规则匹配率: {summary.get('class_rule_found_rate', float('nan')):.4f}\n\n")

        f.write("二、指标说明\n")
        f.write("PSNR/SSIM/LPIPS 为逐样本配对指标，比较原图与该样本匹配到的单条规则可视化图。\n")
        f.write("FID 为分布指标，只在全数据集层面报告一次，不作为单图指标。\n")

    print(f"文本评估报告已保存至: {report_path}")


def _run_decoder_evaluation(run_dir: str, model, decoder, decoder_name: str,
                            eval_loader, eval_meta: Dict[str, Any]):
    decoded_images = _prepare_decoded_rule_images(model, decoder)
    if decoded_images is None:
        print(f"{decoder_name} decoder 未生成规则图片，跳过定量评估。")
        return {}

    eval_dir = run_dir if decoder_name == 'standard' else os.path.join(run_dir, f'evaluation_{decoder_name}')
    os.makedirs(eval_dir, exist_ok=True)
    return _evaluate_rule_visualization_dataset(
        run_dir=eval_dir,
        model=model,
        decoded_images=decoded_images,
        eval_loader=eval_loader,
        eval_meta=eval_meta,
        decoder_name=decoder_name
    )


def _summarize_evaluation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result or not result.get('rows'):
        return {}

    df = pd.DataFrame(result['rows'])
    summary = {
        'decoder_name': result.get('decoder_name', ''),
        'dataset_name': result.get('dataset_name', ''),
        'num_samples': int(result.get('num_samples', len(df))),
        'fid': float(result.get('fid', float('nan'))),
        'class_rule_found_rate': float(df['class_rule_found'].mean()) if 'class_rule_found' in df else float('nan'),
    }

    for metric in ['psnr', 'ssim', 'lpips']:
        values = df[metric].dropna() if metric in df.columns else pd.Series(dtype=float)
        summary[f'{metric}_mean'] = float(values.mean()) if len(values) > 0 else float('nan')
        summary[f'{metric}_std'] = float(values.std()) if len(values) > 0 else float('nan')
        summary[f'{metric}_min'] = float(values.min()) if len(values) > 0 else float('nan')
        summary[f'{metric}_max'] = float(values.max()) if len(values) > 0 else float('nan')

    return summary


def _generate_decoder_comparison_report(run_dir: str, evaluation_results: Dict[str, Dict[str, Any]]):
    summaries = []
    for decoder_name in ['standard', 'gan']:
        summary = _summarize_evaluation_result(evaluation_results.get(decoder_name, {}))
        if summary:
            summaries.append(summary)

    if not summaries:
        print("没有可用于 standard/GAN 对比的评估结果。")
        return

    comparison_df = pd.DataFrame(summaries)
    comparison_csv = os.path.join(run_dir, 'decoder_quality_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
    print(f"standard/GAN 对比汇总已保存至: {comparison_csv}")

    report_path = os.path.join(run_dir, 'decoder_quality_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("Standard vs GAN 规则可视化质量对比报告\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("对比口径: 两个 decoder 使用同一测试集、同一真实类别规则匹配策略、同一指标计算流程。\n")
        f.write("指标方向: PSNR/SSIM 越大越好；LPIPS/FID 越小越好。\n\n")

        metric_cols = ['psnr_mean', 'ssim_mean', 'lpips_mean', 'fid', 'class_rule_found_rate']
        for _, row in comparison_df.iterrows():
            f.write(f"[{row['decoder_name']}]\n")
            for metric in metric_cols:
                value = row.get(metric, float('nan'))
                value_text = 'NaN' if pd.isna(value) else f'{value:.6f}'
                f.write(f"{metric}: {value_text}\n")
            f.write("\n")

        if {'standard', 'gan'}.issubset(set(comparison_df['decoder_name'])):
            std = comparison_df[comparison_df['decoder_name'] == 'standard'].iloc[0]
            gan = comparison_df[comparison_df['decoder_name'] == 'gan'].iloc[0]
            f.write("差值 (GAN - Standard):\n")
            for metric in metric_cols:
                std_v = std.get(metric, float('nan'))
                gan_v = gan.get(metric, float('nan'))
                diff = gan_v - std_v if not (pd.isna(std_v) or pd.isna(gan_v)) else float('nan')
                diff_text = 'NaN' if pd.isna(diff) else f'{diff:.6f}'
                f.write(f"{metric}: {diff_text}\n")
        else:
            f.write("警告: standard 或 gan 中至少一个缺少评估结果，无法计算完整差值。\n")

    print(f"standard/GAN 对比报告已保存至: {report_path}")


def run_visualization(run_dir, compare_decoders=True):
    """Rule visualization plus per-sample quantitative evaluation."""
    print(f"\n>>> 开始规则可视化 (Rule Visualization): {run_dir}")
    print(f">>> 对比模式: {'开启' if compare_decoders else '关闭'}")

    model_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
    configure_model_from_checkpoint(checkpoint)

    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    loaded_decoders = get_decoder_from_config(run_dir, decoder_type='both') or {}
    missing_decoders = [name for name in ['standard', 'gan'] if name not in loaded_decoders]

    if missing_decoders:
        print(f"警告: 未加载到这些 decoder，无法完成完整 standard/GAN 对比: {missing_decoders}")
    else:
        print(f"\n已加载 standard 和 GAN decoder: {list(loaded_decoders.keys())}")

    if compare_decoders and len(loaded_decoders) >= 2:
        visualize_comparison_rules(model, loaded_decoders, run_dir)

    for decoder_type in ['standard', 'gan']:
        decoder = loaded_decoders.get(decoder_type)
        if decoder is not None:
            visualize_single_scale_rules(model, decoder, run_dir, decoder_type=decoder_type)

    if num_rules > 0:
        visualize_attention_heatmap(classifier, num_rules, run_dir)

        consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
        consequents_prob = scipy.special.softmax(consequents, axis=1)
        predicted_classes = np.argmax(consequents_prob, axis=1)
        confidences = np.max(consequents_prob, axis=1)

        rule_specific_attention = None
        if cfg.USE_ATTENTION and classifier.alpha is not None:
            alpha = classifier.alpha[:num_rules].detach()
            rule_specific_attention = F.softmax(alpha, dim=1)
        create_rule_summary_table(run_dir, predicted_classes, confidences, rule_specific_attention)

    if num_rules > 0 and loaded_decoders:
        eval_loader, eval_meta = _load_evaluation_loader(run_dir)
        if eval_loader is not None and eval_meta is not None:
            evaluation_results = {}
            for decoder_name in ['standard', 'gan']:
                decoder = loaded_decoders.get(decoder_name)
                if decoder is None:
                    print(f"跳过 {decoder_name} 定量评估: decoder 未加载。")
                    continue
                evaluation_results[decoder_name] = _run_decoder_evaluation(
                    run_dir, model, decoder, decoder_name, eval_loader, eval_meta
                )
            _generate_decoder_comparison_report(run_dir, evaluation_results)
        else:
            print("未能加载测试集，跳过定量评估。")

    print(f"\n规则可视化完成，结果已保存到: {run_dir}")


if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = 'record\\GTSRB_DFM_FNCN_RESNET18_PRETRAINED_20260408_182253'
    
    
    
    

    if os.path.exists(TEST_DIR):
        run_visualization(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")
