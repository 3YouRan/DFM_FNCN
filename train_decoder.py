import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
import config as cfg
from models import FullModel
import medmnist
from medmnist import BloodMNIST

# 移除硬编码的 RUN_DIR_TO_LOAD

DECODER_EPOCHS = 100
DECODER_LR = 0.001


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


# =========================================================================
# 基础解码器定义 (保持不变)
# =========================================================================
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
    if cfg.USE_MULTI_SCALE_VISUALIZATION and cfg.USE_ATTENTION_GUIDED_DECODER:
        print(f"使用注意力引导的多尺度解码器 (Attention-Guided Multi-Scale Decoder)")
        return AttentionGuidedMultiScaleDecoder(base_class)
    elif cfg.USE_MULTI_SCALE_VISUALIZATION:
        print(f"使用多尺度解码器 (Multi-Scale Decoder)")
        return MultiScaleDecoder(base_class)
    elif cfg.USE_ATTENTION_GUIDED_DECODER:
        print(f"使用注意力引导解码器 (Attention-Guided Decoder)")
        return AttentionGuidedDecoder(base_class)
    else:
        print(f"使用标准解码器 (Standard Decoder)")
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

    data_transform = transforms.Compose([
        transforms.Resize(cfg.TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

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
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


def extract_sample_specific_attention(model, data, features):
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

        # 5. 打印匹配统计信息
        unique_rules, counts = torch.unique(best_rule_idx, return_counts=True)
        print(f"样本-规则匹配统计: {len(unique_rules)}条规则被匹配")
        for rule, count in zip(unique_rules, counts):
            print(f"  规则 {rule.item()}: {count.item()}个样本")

        return batch_attention

def run_decoder_training(run_dir):
    """[修改] 接收 run_dir 参数供 main.py 调用"""
    print(f"\n>>> 开始训练解码器 (Decoder Training): {run_dir}")

    # 检查解码器类型
    if cfg.USE_MULTI_SCALE_VISUALIZATION:
        print(f"多尺度解码器已启用 (Weights: {cfg.MULTI_SCALE_WEIGHTS})")
    elif cfg.USE_ATTENTION_GUIDED_DECODER:
        print(f"注意力引导解码器已启用 (Weight: {cfg.ATTENTION_GUIDED_DECODER_WEIGHT})")
        if not cfg.USE_ATTENTION:
            print("警告: 注意力引导解码器已启用，但模型未使用注意力机制 (USE_ATTENTION=False)")
            print("      解码器将使用默认的注意力权重 (均匀分布)")
    else:
        print("使用标准解码器训练")

    model_path = os.path.join(run_dir, 'best_model.pth')

    # 根据解码器类型选择保存路径
    if cfg.USE_MULTI_SCALE_VISUALIZATION:
        decoder_save_path = os.path.join(run_dir, 'decoder_multi_scale.pth')
    elif cfg.USE_ATTENTION_GUIDED_DECODER:
        decoder_save_path = os.path.join(run_dir, 'decoder_attention_guided.pth')
    else:
        decoder_save_path = os.path.join(run_dir, 'decoder.pth')

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    configure_model_from_checkpoint(checkpoint)

    cfg.MAX_RULES = checkpoint['max_rules']
    print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")

    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    autoencoder = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)
    autoencoder.encoder.requires_grad_(False)
    print("编码器已冻结。开始训练解码器...")

    train_loader = get_train_loader()
    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(DECODER_EPOCHS):
        total_loss = 0
        for batch_idx, data_tuple in enumerate(train_loader):
            data = data_tuple[0].to(cfg.DEVICE)

            # 提取特征
            with torch.no_grad():
                features = base_model.extractor(data)

            # 提取样本特定的注意力权重（如果启用）
            attention_weights = None
            if cfg.USE_ATTENTION_GUIDED_DECODER:
                attention_weights = extract_sample_specific_attention(base_model, data, features)

            optimizer.zero_grad()

            # 根据是否使用注意力引导传递不同的参数
            if cfg.USE_ATTENTION_GUIDED_DECODER and attention_weights is not None:
                reconstructed_images = autoencoder(data, attention_weights)
            else:
                reconstructed_images = autoencoder(data)

            loss = criterion(reconstructed_images, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                loss_info = f"Loss: {loss.item():.6f}"
                if cfg.USE_ATTENTION_GUIDED_DECODER:
                    att_info = " (With Sample-Specific Attention)" if attention_weights is not None else " (No Attention)"
                    loss_info += att_info
                print(f"[Epoch {epoch + 1}/{DECODER_EPOCHS}] Step {batch_idx + 1}/{len(train_loader)} | {loss_info}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n==> Epoch {epoch + 1} 完成. 平均重建损失: {avg_loss:.6f}\n")

    torch.save(autoencoder.decoder.state_dict(), decoder_save_path)
    print(f"解码器训练完成！权重已保存至: {decoder_save_path}")

    # 保存解码器类型信息
    decoder_info = {
        'decoder_type': 'multi_scale' if cfg.USE_MULTI_SCALE_VISUALIZATION else
        ('attention_guided' if cfg.USE_ATTENTION_GUIDED_DECODER else 'standard'),
        'use_attention': cfg.USE_ATTENTION,
        'attention_weight': cfg.ATTENTION_GUIDED_DECODER_WEIGHT if cfg.USE_ATTENTION_GUIDED_DECODER else 0.0,
        'multi_scale_weights': cfg.MULTI_SCALE_WEIGHTS if cfg.USE_MULTI_SCALE_VISUALIZATION else None,
        'attention_method': 'sample_specific'  # 新增：记录注意力方法
    }
    info_path = os.path.join(run_dir, 'decoder_info.pth')
    torch.save(decoder_info, info_path)
    print(f"解码器信息已保存至: {info_path}")

if __name__ == '__main__':
    # 仅用于单独测试
    TEST_DIR = './checkpoints/！！！GTSRB_DFM_FNCN_RESNET18_PRETRAINED_20251209_115250'
    if os.path.exists(TEST_DIR):
        run_decoder_training(TEST_DIR)
    else:
        print("请通过 main.py 运行或在代码中设置有效的 TEST_DIR")