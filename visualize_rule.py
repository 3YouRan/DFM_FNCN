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

    plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)

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

    print(f"\n规则可视化完成！所有结果已保存到: {run_dir}")


if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = 'checkpoints\\MNIST_DFM_FNCN_RESNET18_PRETRAINED_20260130_151401'
    
    

    

    if os.path.exists(TEST_DIR):
        run_visualization(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")