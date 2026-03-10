import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
import time
from datetime import timedelta
from tqdm import tqdm
import config as cfg
from models import FullModel
import medmnist
from medmnist import BloodMNIST


# =========================================================================
# CBAM 注意力模块 (用于解码器)
# =========================================================================
class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))


# =========================================================================
# CBAM 注意力模块 (用于解码器)
# =========================================================================
class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))


class CBAMDecoder(nn.Module):
    """CBAM 注意力引导解码器：结合通道注意力和空间注意力"""
    def __init__(self, base_decoder_class, reduction=16, kernel_size=7):
        super(CBAMDecoder, self).__init__()
        self.base_decoder = base_decoder_class()
        
        # CBAM 注意力模块
        self.channel_attention = ChannelAttention(cfg.N_CHANNELS_OUT, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # 规则注意力投影层 (用于规则级别的注意力权重)
        self.rule_attention_projection = nn.Sequential(
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT)
        )
        
    def forward(self, x, attention_weights=None):
        """
        x: 规则中心特征图 (B, C, H, W)
        attention_weights: 规则级别的注意力权重 (B, C) 或 None
        """
        # 首先应用 CBAM 注意力
        x = self._apply_cbam(x)
        
        # 然后应用规则级别的注意力引导 (如果提供)
        if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
            x = self._apply_rule_attention(x, attention_weights)
        
        # 通过基础解码器
        return self.base_decoder(x)
    
    def _apply_cbam(self, x):
        """应用 CBAM 注意力"""
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x
    
    def _apply_rule_attention(self, x, attention_weights):
        """应用规则级别的注意力权重"""
        # 将注意力权重投影到特征维度
        att_projected = self.rule_attention_projection(attention_weights)
        
        # 扩展到特征图维度
        B, C, H, W = x.shape
        att_expanded = att_projected.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        # 调制特征
        modulated = x * (1 + cfg.ATTENTION_GUIDED_DECODER_WEIGHT * att_expanded)
        return modulated

# [新增] 导入混合精度训练模块
try:
    from torch.cuda.amp import  GradScaler,autocast 
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: CUDA AMP 不可用，将使用标准精度训练")

# 移除硬编码的 RUN_DIR_TO_LOAD

DECODER_EPOCHS = 300
DECODER_LR = 0.001
BATCH_SIZE = 256

def configure_model_from_checkpoint(checkpoint):
    """[新] 从检查点中的 'config_params' 推断配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    if params['MODEL_TYPE'] != 'DFM_FNCN':
        print("错误: 解码器只能为 DFM_FNCN 模型训练。")
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


# =========================================================================
# 基础解码器模块 (保持不变)
# =========================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out


# =========================================================================
# 创新点1：注意力引导的解码器 (Attention-Guided Decoder)
# =========================================================================
class AttentionGuidedDecoder(nn.Module):
    def __init__(self, base_decoder_class):
        """
        注意力引导解码器包装器
        base_decoder_class: 基础解码器类 (SimpleCNNDecoder, ResNet18Decoder, VGG16Decoder)
        """
        super(AttentionGuidedDecoder, self).__init__()
        self.base_decoder = base_decoder_class()

        # 注意力调制层：将特征和注意力权重融合
        self.attention_modulation = nn.Sequential(
            nn.Conv2d(cfg.N_CHANNELS_OUT * 2, cfg.N_CHANNELS_OUT, kernel_size=1),
            nn.BatchNorm2d(cfg.N_CHANNELS_OUT),
            nn.ReLU(inplace=True)
        )

        # 注意力权重投影层：将规则级别的注意力映射到特征图级别
        self.attention_projection = nn.Sequential(
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT)
        )

    def forward(self, x, attention_weights=None):
        """
        x: 规则中心特征图 (B, C, H, W)
        attention_weights: 注意力权重 (B, C) 或 None
        """
        if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
            # 步骤1: 将注意力权重投影到合适的维度
            # attention_weights: (B, C) -> 经过投影 -> (B, C)
            att_projected = self.attention_projection(attention_weights)

            # 步骤2: 将注意力权重扩展到特征图维度
            # (B, C) -> (B, C, 1, 1) -> (B, C, H, W)
            B, C, H, W = x.shape
            att_expanded = att_projected.view(B, C, 1, 1)
            att_expanded = att_expanded.expand(-1, -1, H, W)

            # 步骤3: 使用注意力权重调制特征
            # 应用调制强度
            att_modulated = att_expanded * cfg.ATTENTION_GUIDED_DECODER_WEIGHT

            # 步骤4: 拼接原始特征和注意力调制特征
            combined = torch.cat([x, att_modulated], dim=1)

            # 步骤5: 通过调制层融合
            modulated_features = self.attention_modulation(combined)

            # 步骤6: 使用基础解码器解码
            return self.base_decoder(modulated_features)
        else:
            # 如果没有注意力权重或未启用注意力引导，使用原始解码器
            return self.base_decoder(x)


class SimpleCNNDecoder(nn.Module):
    def __init__(self):
        super(SimpleCNNDecoder, self).__init__()
        self.bottleneck = nn.Sequential(
            BasicBlock(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT),
            BasicBlock(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS_OUT, 16, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, cfg.IN_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


class ResNet18Decoder(nn.Module):
    def __init__(self):
        super(ResNet18Decoder, self).__init__()
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS_OUT, 256, kernel_size=1, bias=False)
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256), BasicBlock(256, 256),
            BasicBlock(256, 256), BasicBlock(256, 256)
        )
        self.reverse_pool = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1)
        self.reverse_layer3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev3 = nn.BatchNorm2d(128)
        self.reverse_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev2 = nn.BatchNorm2d(64)
        self.reverse_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)
        x = F.relu(self.reverse_pool(x))
        x = F.relu(self.bn_rev3(self.reverse_layer3(x)))
        x = F.relu(self.bn_rev2(self.reverse_layer2(x)))
        x = self.reverse_conv1(x)
        return x


class VGG16Decoder(nn.Module):
    def __init__(self):
        super(VGG16Decoder, self).__init__()
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS_OUT, 256, kernel_size=1, bias=False)
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256), BasicBlock(256, 256),
            BasicBlock(256, 256), BasicBlock(256, 256)
        )
        self.reverse_pool = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1)
        self.reverse_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.reverse_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.reverse_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)
        x = F.relu(self.reverse_pool(x))
        x = self.reverse_block3(x)
        x = F.relu(self.upsample1(x))
        x = self.reverse_block2(x)
        x = F.relu(self.upsample2(x))
        x = self.reverse_block1(x)
        return x


# =========================================================================
# 创新点2：多尺度解码器 (Multi-Scale Decoder)
# =========================================================================
# =========================================================================
# 创新点2：多尺度解码器 (Multi-Scale Decoder) - 修复版本
# =========================================================================
class MultiScaleDecoder(nn.Module):
    def __init__(self, base_decoder_class):
        """
        多尺度解码器：生成不同尺度的重建图像
        """
        super(MultiScaleDecoder, self).__init__()

        # 基础解码器（用于完整分辨率）
        self.base_decoder = base_decoder_class()

        # 多尺度解码分支
        self.coarse_decoder = self._create_coarse_decoder()
        self.medium_decoder = self._create_medium_decoder()
        self.fine_decoder = self._create_fine_decoder()

        # 尺度融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS * 3, cfg.IN_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(cfg.IN_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.IN_CHANNELS, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 目标输出尺寸
        self.target_size = cfg.TARGET_SIZE

    def _create_coarse_decoder(self):
        """粗尺度解码器：关注整体形状"""
        return nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS_OUT, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _create_medium_decoder(self):
        """中尺度解码器：关注主要特征"""
        return nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS_OUT, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _create_fine_decoder(self):
        """细尺度解码器：关注细节纹理"""
        return nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS_OUT, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, attention_weights=None, scale=None):
        """
        x: 特征图 (B, C, H, W)
        attention_weights: 注意力权重 (B, C) 或 None
        scale: 指定尺度 ('coarse', 'medium', 'fine', 'all', None)
        """
        # 注意力调制（如果启用）
        if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
            B, C, H, W = x.shape
            att_expanded = attention_weights.view(B, C, 1, 1).expand(-1, -1, H, W)
            x = x * (1 + cfg.ATTENTION_GUIDED_DECODER_WEIGHT * att_expanded)

        # 根据指定尺度返回相应输出
        if scale == 'coarse':
            output = self.coarse_decoder(x)
            # 上采样到目标尺寸
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'medium':
            output = self.medium_decoder(x)
            # 上采样到目标尺寸
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'fine':
            output = self.fine_decoder(x)
            # 上采样到目标尺寸
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'all' or scale is None:
            # 生成所有尺度并融合
            coarse = self.coarse_decoder(x)
            medium = self.medium_decoder(x)
            fine = self.fine_decoder(x)

            # 上采样到相同尺寸（目标尺寸）
            coarse_up = F.interpolate(coarse, size=self.target_size, mode='bilinear', align_corners=False)
            medium_up = F.interpolate(medium, size=self.target_size, mode='bilinear', align_corners=False)
            fine_up = F.interpolate(fine, size=self.target_size, mode='bilinear', align_corners=False)

            # 加权融合
            fused = torch.cat([
                coarse_up * cfg.MULTI_SCALE_WEIGHTS[0],
                medium_up * cfg.MULTI_SCALE_WEIGHTS[1],
                fine_up * cfg.MULTI_SCALE_WEIGHTS[2]
            ], dim=1)

            output = self.fusion_layer(fused)
            return output
        else:
            # 默认使用基础解码器
            return self.base_decoder(x)


# =========================================================================
# 创新点组合：注意力引导的多尺度解码器
# =========================================================================
class AttentionGuidedMultiScaleDecoder(nn.Module):
    def __init__(self, base_decoder_class):
        """
        注意力引导的多尺度解码器：结合两个创新点
        """
        super(AttentionGuidedMultiScaleDecoder, self).__init__()

        # 基础解码器（用于完整分辨率）
        self.base_decoder = base_decoder_class()

        # 注意力调制层
        self.attention_modulation = nn.Sequential(
            nn.Conv2d(cfg.N_CHANNELS_OUT * 2, cfg.N_CHANNELS_OUT, kernel_size=1),
            nn.BatchNorm2d(cfg.N_CHANNELS_OUT),
            nn.ReLU(inplace=True)
        )

        # 注意力权重投影层
        self.attention_projection = nn.Sequential(
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT)
        )

        # 多尺度解码分支
        self.coarse_decoder = self._create_coarse_decoder() # pyright: ignore[reportCallIssue]
        self.medium_decoder = self._create_medium_decoder() # pyright: ignore[reportCallIssue]
        self.fine_decoder = self._create_fine_decoder() # pyright: ignore[reportCallIssue]

        # 尺度融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS * 3, cfg.IN_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(cfg.IN_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.IN_CHANNELS, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 目标输出尺寸
        self.target_size = cfg.TARGET_SIZE

    def _apply_attention_modulation(self, x, attention_weights):
        """应用注意力调制（支持批量或单个规则）"""
        if attention_weights is not None:
            # 步骤1: 将注意力权重投影到合适的维度
            att_projected = self.attention_projection(attention_weights)

            # 步骤2: 将注意力权重扩展到特征图维度
            B, C, H, W = x.shape
            att_expanded = att_projected.view(B, C, 1, 1)
            att_expanded = att_expanded.expand(-1, -1, H, W)

            # 步骤3: 使用注意力权重调制特征
            att_modulated = att_expanded * cfg.ATTENTION_GUIDED_DECODER_WEIGHT

            # 步骤4: 拼接原始特征和注意力调制特征
            combined = torch.cat([x, att_modulated], dim=1)

            # 步骤5: 通过调制层融合
            return self.attention_modulation(combined)
        else:
            return x

    def forward(self, x, attention_weights=None, scale=None):
        """
        x: 特征图 (B, C, H, W)
        attention_weights: 注意力权重 (B, C) 或 None
        scale: 指定尺度 ('coarse', 'medium', 'fine', 'all', None)
        """
        # 应用注意力调制
        if attention_weights is not None:
            x = self._apply_attention_modulation(x, attention_weights)

        # 根据指定尺度返回相应输出
        if scale == 'coarse':
            output = self.coarse_decoder(x)
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'medium':
            output = self.medium_decoder(x)
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'fine':
            output = self.fine_decoder(x)
            output = F.interpolate(output, size=self.target_size, mode='bilinear', align_corners=False)
            return output
        elif scale == 'all' or scale is None:
            # 生成所有尺度并融合
            coarse = self.coarse_decoder(x)
            medium = self.medium_decoder(x)
            fine = self.fine_decoder(x)

            # 上采样到相同尺寸（目标尺寸）
            coarse_up = F.interpolate(coarse, size=self.target_size, mode='bilinear', align_corners=False)
            medium_up = F.interpolate(medium, size=self.target_size, mode='bilinear', align_corners=False)
            fine_up = F.interpolate(fine, size=self.target_size, mode='bilinear', align_corners=False)

            # 加权融合
            fused = torch.cat([
                coarse_up * cfg.MULTI_SCALE_WEIGHTS[0],
                medium_up * cfg.MULTI_SCALE_WEIGHTS[1],
                fine_up * cfg.MULTI_SCALE_WEIGHTS[2]
            ], dim=1)

            output = self.fusion_layer(fused)
            return output
        else:
            # 默认使用基础解码器
            return self.base_decoder(x)

# =========================================================================
# 创新点：GAN解码器 (GAN Decoder)
# =========================================================================
class Discriminator(nn.Module):
    """判别器网络，输入图像，输出真实/虚假概率"""
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # 简单卷积网络
        self.model = nn.Sequential(
            # 输入: (in_channels, H, W)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, img):
        # 输入 img: (B, C, H, W)
        validity = self.model(img)
        return validity.view(-1, 1)

class GANDecoder(nn.Module):
    """GAN解码器：包含生成器（基础解码器）和判别器"""
    def __init__(self, base_decoder_class):
        super(GANDecoder, self).__init__()
        self.generator = base_decoder_class()
        self.discriminator = Discriminator(cfg.IN_CHANNELS)

    def forward(self, x, attention_weights=None):
        """
        生成器前向传播，返回重建图像
        x: 特征图 (B, C, H, W)
        attention_weights: 注意力权重 (B, C) 或 None (如果基础解码器支持)
        """
        # 如果基础解码器支持注意力权重，传递给它
        if hasattr(self.generator, 'forward') and callable(getattr(self.generator, 'forward', None)):
            # 检查基础解码器是否接受attention_weights参数
            import inspect
            sig = inspect.signature(self.generator.forward)
            params = list(sig.parameters.keys())
            if 'attention_weights' in params:
                return self.generator(x, attention_weights=attention_weights)
            else:
                return self.generator(x)
        else:
            return self.generator(x)

    def discriminate(self, img):
        """判别器前向传播，返回真实/虚假分数"""
        return self.discriminator(img)
def get_decoder():
    """获取解码器，根据配置选择类型"""
    # 确定基础解码器类
    if cfg.EXTRACTOR_TYPE == 'RESNET18_PRETRAINED':
        base_class = ResNet18Decoder
    elif cfg.EXTRACTOR_TYPE == 'VGG16_PRETRAINED':
        base_class = VGG16Decoder
    elif cfg.EXTRACTOR_TYPE == 'SIMPLE_CNN':
        base_class = SimpleCNNDecoder
    else:
        raise ValueError(f"未知的 EXTRACTOR_TYPE: {cfg.EXTRACTOR_TYPE}")

    # 根据配置选择解码器类型
    if cfg.USE_GAN_DECODER:
        decoder_type = "GAN解码器"
        decoder_info_str = f"  Adversarial Weight: {cfg.GAN_ADVERSARIAL_WEIGHT}"
        if cfg.GAN_USE_LSGAN:
            decoder_info_str += ", 损失: LSGAN"
    elif cfg.USE_ATTENTION_GUIDED_DECODER and hasattr(cfg, 'ATTENTION_TYPE') and cfg.ATTENTION_TYPE == 'CBAM':
        # CBAM 解码器
        decoder_type = "CBAM注意力解码器"
        decoder_info_str = f"  CBAM Reduction: {cfg.CBAM_REDUCTION}, Kernel Size: {cfg.CBAM_KERNEL_SIZE}"
    elif cfg.USE_MULTI_SCALE_VISUALIZATION and cfg.USE_ATTENTION_GUIDED_DECODER:
        decoder_type = "注意力引导多尺度解码器"
        decoder_info_str = f"  权重: {cfg.MULTI_SCALE_WEIGHTS}"
    elif cfg.USE_MULTI_SCALE_VISUALIZATION:
        decoder_type = "多尺度解码器"
        decoder_info_str = f"  权重: {cfg.MULTI_SCALE_WEIGHTS}"
    elif cfg.USE_ATTENTION_GUIDED_DECODER:
        decoder_type = "注意力引导解码器"
        decoder_info_str = f"  Weight: {cfg.ATTENTION_GUIDED_DECODER_WEIGHT}"
    else:
        decoder_type = "标准解码器"
        decoder_info_str = ""

    print(f"解码器类型: {decoder_type}")
    if decoder_info_str:
        print(decoder_info_str)
    if cfg.USE_ATTENTION_GUIDED_DECODER and not cfg.USE_ATTENTION:
        print("  警告: 解码器已启用注意力引导，但模型未使用注意力机制")

    # 返回相应的解码器
    if cfg.USE_GAN_DECODER:
        return GANDecoder(base_class)
    elif cfg.USE_ATTENTION_GUIDED_DECODER and hasattr(cfg, 'ATTENTION_TYPE') and cfg.ATTENTION_TYPE == 'CBAM':
        return CBAMDecoder(base_class, reduction=cfg.CBAM_REDUCTION, kernel_size=cfg.CBAM_KERNEL_SIZE)
    elif cfg.USE_MULTI_SCALE_VISUALIZATION and cfg.USE_ATTENTION_GUIDED_DECODER:
        return AttentionGuidedMultiScaleDecoder(base_class)
    elif cfg.USE_MULTI_SCALE_VISUALIZATION:
        return MultiScaleDecoder(base_class)
    elif cfg.USE_ATTENTION_GUIDED_DECODER:
        return AttentionGuidedDecoder(base_class)
    else:
        return base_class()
class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = get_decoder()

    def forward(self, x, attention_weights=None):
        with torch.no_grad():
            features = self.encoder(x)

        # 根据解码器类型传递不同的参数
        if cfg.USE_ATTENTION_GUIDED_DECODER:
            reconstruction = self.decoder(features, attention_weights)
        else:
            reconstruction = self.decoder(features)

        return reconstruction


def get_train_loader():
    if cfg.IN_CHANNELS == 1:
        norm_mean, norm_std = (0.5,), (0.5,)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # 构建转换列表
    transform_list = [transforms.Resize(cfg.TARGET_SIZE)]
    # 对于 GEOMETRIC_SHAPES 和 SHAPES_CLASSIFICATION 数据集，添加灰度转换
    if cfg.DATASET_NAME == 'GEOMETRIC_SHAPES' or cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION':
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    data_transform = transforms.Compose(transform_list)

    if cfg.DATASET_NAME == 'FASHION_MNIST':
        train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'SVHN':
        train_dataset = datasets.SVHN(root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform)
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        train_dataset = BloodMNIST(split='train', transform=data_transform, download=True, root=cfg.DATA_ROOT)
    elif cfg.DATASET_NAME == 'GTSRB':
        # [修改] GTSRB 子集处理逻辑
        target_transform = None
        if cfg.GTSRB_SUBSET_INDICES is not None:
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(cfg.GTSRB_SUBSET_INDICES)}
            target_transform = transforms.Lambda(lambda y: mapping.get(y, -1))

        train_dataset = datasets.GTSRB(root=cfg.DATA_ROOT, split='train', download=True,
                                       transform=data_transform, target_transform=target_transform)

        if cfg.GTSRB_SUBSET_INDICES is not None:
            subset_set = set(cfg.GTSRB_SUBSET_INDICES)
            train_indices = [i for i, (_, label) in enumerate(train_dataset._samples) if label in subset_set]
            train_dataset = Subset(train_dataset, train_indices)
    
    elif cfg.DATASET_NAME == 'MNIST':
        train_dataset = datasets.MNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
    
    elif cfg.DATASET_NAME == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
    
    elif cfg.DATASET_NAME == 'CIFAR100':
        # CIFAR100 子集处理逻辑
        target_transform = None
        
        # 确定要使用的子集索引
        selected_indices = None
        if cfg.CIFAR100_SUBSET_NAMES is not None:
            # 将类别名称转换为索引
            selected_indices = []
            for name in cfg.CIFAR100_SUBSET_NAMES:
                if name in cfg.CIFAR100_ALL_CLASSES:
                    selected_indices.append(cfg.CIFAR100_ALL_CLASSES.index(name))
                else:
                    raise ValueError(f"未知的 CIFAR100 类别名称: {name}")
        elif cfg.CIFAR100_SUBSET_INDICES is not None:
            selected_indices = cfg.CIFAR100_SUBSET_INDICES
        
        train_dataset = datasets.CIFAR100(root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform)
        
        if selected_indices is not None:
            # 1. 创建标签映射: 原始ID -> 0..N-1
            mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
            # 2. 过滤数据集
            subset_set = set(selected_indices)
            train_indices = []
            
            # 训练集过滤
            train_targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.targets
            for i, label in enumerate(train_targets):
                if label in subset_set:
                    train_indices.append(i)
            
            train_dataset = Subset(train_dataset, train_indices)
            print(f"CIFAR100 Subset: Train {len(train_dataset)}")
    
    elif cfg.DATASET_NAME == 'GEOMETRIC_SHAPES':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'geometric_shapes'), transform=data_transform)
        # 按与训练相同的比例分割（训练集占80%）
        train_ratio = 0.8
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Geometric Shapes 训练集大小: {len(train_dataset)}")
    
    elif cfg.DATASET_NAME == 'MIO_TCD_CLASSIFICATION':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'MIO-TCD-Classification'), transform=data_transform)
        # 按与训练相同的比例分割（训练集占4/5）
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"MIO-TCD-Classification 训练集大小: {len(train_dataset)}")
    
    elif cfg.DATASET_NAME == 'VEHICLES':
        # 加载整个数据集（无预定义分割）
        full_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATA_ROOT, 'Vehicles'), transform=data_transform)
        # 按与训练相同的比例分割（训练集占4/5）
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Vehicles 训练集大小: {len(train_dataset)}")
    
    elif cfg.DATASET_NAME == 'SHAPES_CLASSIFICATION':
        # 加载 Shapes Classification 数据集
        # 路径: data/Shapes_Classification/archive(6)/shapes/
        dataset_path = os.path.join(cfg.DATA_ROOT, 'Shapes_Classification', 'archive(6)', 'shapes')
        full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
        # 按与训练相同的比例分割（训练集占4/5）
        train_ratio = 4/5
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.SEED))
        print(f"Shapes Classification 训练集大小: {len(train_dataset)}")
    
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0)
    return train_loader


def extract_sample_specific_attention(model, data, features, print_attention_stats=False):
    """
    [改进] 为每个样本分配其最匹配规则的注意力权重
    返回: (B, C) 每个样本使用其最匹配规则的注意力权重
    """
    if not cfg.USE_ATTENTION or not cfg.USE_ATTENTION_GUIDED_DECODER:
        return None

    model.eval()
    with torch.no_grad():
        # 获取分类器中的注意力权重
        classifier = model.classifier
        num_rules = classifier.num_active_rules.item()

        if num_rules == 0 or classifier.alpha is None:
            return None

        # 1. 获取所有规则的注意力权重 (Rules, Channels)
        all_att_weights = F.softmax(classifier.alpha[:num_rules], dim=1)

        # 2. 计算每个样本与所有规则的匹配度
        # 使用分类器的前向传播逻辑计算phi
        b = features.size(0)
        x = model.classifier.bn(features)
        x_flat = x.view(b, model.classifier.n_channels, -1)

        active_centers = classifier.centers[:num_rules]
        active_widths_param = classifier.widths_param[:num_rules]

        # 计算隶属度
        x_exp, c_exp = x_flat.unsqueeze(1), active_centers.unsqueeze(0)
        M = F.cosine_similarity(x_exp, c_exp, dim=3)
        d = 1.0 - M
        sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
        mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

        # 计算激发强度phi
        if cfg.USE_ATTENTION:
            # 使用注意力权重计算加权乘积
            att_weights = all_att_weights.unsqueeze(0)  # (1, Rules, Channels)
            log_mu = torch.log(mu.to(torch.float64) + 1e-9)
            weighted_log_sum = torch.sum(att_weights.to(torch.float64) * log_mu, dim=2) * model.classifier.n_channels
            phi_double = torch.exp(weighted_log_sum)
        else:
            phi_double = torch.prod(mu.to(torch.float64), dim=2)

        phi = phi_double.to(torch.float32)

        # 3. 为每个样本选择最匹配的规则
        # 找到每个样本激活最强的规则
        best_rule_idx = torch.argmax(phi, dim=1)  # (B,)

        # 4. 获取对应规则的注意力权重
        batch_attention = all_att_weights[best_rule_idx]  # (B, Channels)

        # 5. 打印匹配统计信息（使用标志控制，避免重复输出）
        if print_attention_stats:
            unique_rules, counts = torch.unique(best_rule_idx, return_counts=True)
            print(f"  样本-规则匹配: {len(unique_rules)}条规则被激活")
            for rule, count in zip(unique_rules, counts):
                print(f"    规则 {rule.item()}: {count.item()}个样本")

        return batch_attention

def train_standard_decoder(autoencoder, base_model, train_loader, decoder_save_path, use_amp, scaler):
    """训练标准解码器（普通解码器，不带GAN）"""
    print("\n" + "="*60)
    print("训练标准解码器（普通MSE重建）")
    print("="*60)
    
    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()
    
    autoencoder.train()
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(DECODER_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Standard Epoch {epoch+1}/{DECODER_EPOCHS}",
                   ncols=100, unit="batch", leave=False, dynamic_ncols=False)
        
        for batch_idx, data_tuple in pbar:
            data = data_tuple[0].to(cfg.DEVICE)
            
            if data.size(0) == 0:
                continue
            
            with torch.no_grad():
                features = base_model.extractor(data)
            
            attention_weights = None
            if cfg.USE_ATTENTION_GUIDED_DECODER:
                attention_weights = extract_sample_specific_attention(
                    base_model, data, features, print_attention_stats=(batch_idx == 0)
                )
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(): # pyright: ignore[reportPossiblyUnboundVariable]
                    if cfg.USE_ATTENTION_GUIDED_DECODER and attention_weights is not None:
                        reconstructed_images = autoencoder.decoder(features, attention_weights)
                    else:
                        reconstructed_images = autoencoder.decoder(features)
                    loss = criterion(reconstructed_images, data)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.USE_ATTENTION_GUIDED_DECODER and attention_weights is not None:
                    reconstructed_images = autoencoder.decoder(features, attention_weights)
                else:
                    reconstructed_images = autoencoder.decoder(features)
                loss = criterion(reconstructed_images, data)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(autoencoder.decoder.state_dict(), decoder_save_path)
            print(f"  *** Best Standard Decoder (Loss: {best_loss:.4f}) ***")
        
        print(f"Epoch {epoch + 1}/{DECODER_EPOCHS} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Best: Epoch {best_epoch} (Loss: {best_loss:.4f})")
    
    print(f"\n标准解码器训练完成！最佳权重在 Epoch {best_epoch} (Loss: {best_loss:.4f})")
    print(f"权重已保存至: {decoder_save_path}")
    return best_loss, best_epoch


def train_gan_decoder(autoencoder, base_model, train_loader, decoder_save_path, discriminator_save_path, use_amp, scaler):
    """训练GAN解码器（带对抗损失）"""
    print("\n" + "="*60)
    print("训练GAN解码器（带对抗损失）")
    print("="*60)
    print(f"Adversarial Weight: {cfg.GAN_ADVERSARIAL_WEIGHT}")
    if cfg.GAN_USE_LSGAN:
        print("使用LSGAN损失")
    
    generator = autoencoder.decoder.generator
    discriminator = autoencoder.decoder.discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=cfg.GAN_GENERATOR_LR)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.GAN_DISCRIMINATOR_LR)
    criterion_recon = nn.MSELoss()
    
    if cfg.GAN_USE_LSGAN:
        criterion_adv = nn.MSELoss()
    else:
        criterion_adv = nn.BCEWithLogitsLoss()
    
    autoencoder.train()
    best_loss_G = float('inf')
    best_epoch = 0
    
    for epoch in range(DECODER_EPOCHS):
        epoch_start_time = time.time()
        total_loss_G = 0
        total_loss_D = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f"GAN Epoch {epoch+1}/{DECODER_EPOCHS}",
                   ncols=100, unit="batch", leave=False, dynamic_ncols=False)
        
        for batch_idx, data_tuple in pbar:
            data = data_tuple[0].to(cfg.DEVICE)
            
            if data.size(0) == 0:
                continue
            
            with torch.no_grad():
                features = base_model.extractor(data)
            
            attention_weights = None
            if cfg.USE_ATTENTION_GUIDED_DECODER:
                attention_weights = extract_sample_specific_attention(
                    base_model, data, features, print_attention_stats=(batch_idx == 0)
                )
            
            batch_size = data.size(0)
            real_label = torch.ones(batch_size, 1, device=cfg.DEVICE)
            fake_label = torch.zeros(batch_size, 1, device=cfg.DEVICE)
            
            # 训练判别器
            optimizer_D.zero_grad()
            
            if use_amp:
                with autocast(): # pyright: ignore[reportPossiblyUnboundVariable]
                    real_pred = discriminator(data)
                    loss_D_real = criterion_adv(real_pred, real_label)
                    
                    if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
                        fake_images = autoencoder.decoder(features, attention_weights)
                    else:
                        fake_images = autoencoder.decoder(features)
                    fake_pred = discriminator(fake_images.detach())
                    loss_D_fake = criterion_adv(fake_pred, fake_label)
                    
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
            else:
                real_pred = discriminator(data)
                loss_D_real = criterion_adv(real_pred, real_label)
                
                if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
                    fake_images = autoencoder.decoder(features, attention_weights)
                else:
                    fake_images = autoencoder.decoder(features)
                fake_pred = discriminator(fake_images.detach())
                loss_D_fake = criterion_adv(fake_pred, fake_label)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
            
            # 训练生成器
            optimizer_G.zero_grad()
            
            if use_amp:
                with autocast(): # pyright: ignore[reportPossiblyUnboundVariable]
                    if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
                        fake_images = autoencoder.decoder(features, attention_weights)
                    else:
                        fake_images = autoencoder.decoder(features)
                    loss_recon = criterion_recon(fake_images, data)
                    
                    fake_pred = discriminator(fake_images)
                    loss_adv = criterion_adv(fake_pred, real_label)
                    
                    loss_G = loss_recon + cfg.GAN_ADVERSARIAL_WEIGHT * loss_adv
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                if attention_weights is not None and cfg.USE_ATTENTION_GUIDED_DECODER:
                    fake_images = autoencoder.decoder(features, attention_weights)
                else:
                    fake_images = autoencoder.decoder(features)
                loss_recon = criterion_recon(fake_images, data)
                
                fake_pred = discriminator(fake_images)
                loss_adv = criterion_adv(fake_pred, real_label)
                
                loss_G = loss_recon + cfg.GAN_ADVERSARIAL_WEIGHT * loss_adv
                loss_G.backward()
                optimizer_G.step()
            
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            
            pbar.set_postfix({'G': f'{loss_G.item():.3f}', 'D': f'{loss_D.item():.3f}'})
        
        avg_loss_G = total_loss_G / len(train_loader)
        avg_loss_D = total_loss_D / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            best_epoch = epoch + 1
            torch.save(generator.state_dict(), decoder_save_path)
            if discriminator_save_path:
                torch.save(discriminator.state_dict(), discriminator_save_path)
            print(f"  *** Best GAN Generator (Loss: {best_loss_G:.4f}) ***")
        
        print(f"Epoch {epoch + 1}/{DECODER_EPOCHS} | Loss_G: {avg_loss_G:.4f} | Loss_D: {avg_loss_D:.4f} | Time: {epoch_time:.1f}s | Best: Epoch {best_epoch} (Loss: {best_loss_G:.4f})")
    
    print(f"\nGAN解码器训练完成！最佳权重在 Epoch {best_epoch} (Loss: {best_loss_G:.4f})")
    print(f"生成器权重已保存至: {decoder_save_path}")
    if discriminator_save_path:
        print(f"判别器权重已保存至: {discriminator_save_path}")
    
    return best_loss_G, best_epoch


def run_decoder_training(run_dir):
    """[修改] 支持训练两种解码器进行对比实验"""
    print(f"\n{'='*60}")
    print(f"开始训练解码器（对比实验模式）")
    print(f"{'='*60}")
    
    total_start_time = time.time()
    
    model_path = os.path.join(run_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return
    
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
    configure_model_from_checkpoint(checkpoint)
    cfg.MAX_RULES = checkpoint['max_rules']
    print(f"配置: MAX_RULES={cfg.MAX_RULES}, 训练轮数={DECODER_EPOCHS}")
    
    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存路径定义
    gan_decoder_path = os.path.join(run_dir, 'decoder_gan.pth')
    gan_discriminator_path = os.path.join(run_dir, 'discriminator.pth')
    standard_decoder_path = os.path.join(run_dir, 'decoder_standard.pth')
    
    # [新增] 初始化混合精度训练
    use_amp = AMP_AVAILABLE and cfg.DEVICE.type == 'cuda'
    if use_amp:
        print("混合精度训练已启用 (AMP)")
        scaler = GradScaler() # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        print("使用标准精度训练")
        scaler = None
    
    train_loader = get_train_loader()
    
    # 训练结果记录
    training_results = {}
    
    # 1. 训练标准解码器（普通MSE重建）
    print("\n" + "="*60)
    print("开始训练标准解码器...")
    print("="*60)
    
    # 创建标准解码器的Autoencoder
    original_use_gan = cfg.USE_GAN_DECODER
    cfg.USE_GAN_DECODER = False  # 临时禁用GAN，使用标准解码器
    autoencoder_std = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)
    autoencoder_std.encoder.requires_grad_(False)
    print("编码器已冻结。开始训练标准解码器...")
    
    best_loss_std, best_epoch_std = train_standard_decoder(
        autoencoder_std, base_model, train_loader, 
        standard_decoder_path, use_amp, scaler
    )
    training_results['standard'] = {'best_loss': best_loss_std, 'best_epoch': best_epoch_std}
    
    # 恢复配置
    cfg.USE_GAN_DECODER = original_use_gan
    
    # 2. 训练GAN解码器（带对抗损失）
    print("\n" + "="*60)
    print("开始训练GAN解码器...")
    print("="*60)
    
    # 创建GAN解码器的Autoencoder
    cfg.USE_GAN_DECODER = True  # 启用GAN
    autoencoder_gan = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)
    autoencoder_gan.encoder.requires_grad_(False)
    print("编码器已冻结。开始训练GAN解码器...")
    
    best_loss_gan, best_epoch_gan = train_gan_decoder(
        autoencoder_gan, base_model, train_loader,
        gan_decoder_path, gan_discriminator_path, use_amp, scaler
    )
    training_results['gan'] = {'best_loss': best_loss_gan, 'best_epoch': best_epoch_gan}
    
    # 恢复配置
    cfg.USE_GAN_DECODER = original_use_gan
    
    # 3. 打印对比结果
    print("\n" + "="*60)
    print("解码器训练对比结果")
    print("="*60)
    print(f"标准解码器 - 最佳Loss: {best_loss_std:.4f} (Epoch {best_epoch_std})")
    print(f"GAN解码器   - 最佳Loss: {best_loss_gan:.4f} (Epoch {best_epoch_gan})")
    print("="*60)
    
    # 保存训练结果信息
    results_path = os.path.join(run_dir, 'decoder_comparison_results.pth')
    torch.save(training_results, results_path)
    print(f"对比结果已保存至: {results_path}")
    
    # 保存解码器信息
    decoder_info = {
        'standard_decoder_path': standard_decoder_path,
        'gan_decoder_path': gan_decoder_path,
        'gan_discriminator_path': gan_discriminator_path,
        'training_results': training_results
    }
    info_path = os.path.join(run_dir, 'decoder_info.pth')
    torch.save(decoder_info, info_path)
    print(f"解码器信息已保存至: {info_path}")
    
    total_time = time.time() - total_start_time
    print(f"\n解码器训练完成！总耗时: {str(timedelta(seconds=int(total_time)))}")

if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = 'record\\MNIST_DFM_FNCN_RESNET18_PRETRAINED_20260305_223501'
    if os.path.exists(TEST_DIR):
        run_decoder_training(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")
