import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg


# =========================================================================
# 改进的特征提取器 (ResNet-like)
# =========================================================================

class BasicBlock(nn.Module):
    """ResNet的基本残差块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 如果输入输出维度不一致，使用 1x1 卷积进行匹配
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFeatureExtractor(nn.Module):
    """
    专为 Fashion-MNIST (28x28) 设计的小型 ResNet 特征提取器。
    它保证输出维度严格匹配 config.py 中的设置: [Batch, N_CHANNELS(32), 7, 7]
    """

    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.in_planes = 64

        # 初始层: 1x28x28 -> 64x28x28
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 1: 保持 28x28
        self.layer1 = self._make_layer(64, 2, stride=1)
        # Layer 2: 下采样到 14x14
        self.layer2 = self._make_layer(128, 2, stride=2)
        # Layer 3: 下采样到 7x7
        self.layer3 = self._make_layer(256, 2, stride=2)

        # 最终投影层: 将 256 通道映射回我们 config 中需要的 32 通道 (cfg.N_CHANNELS)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 * BasicBlock.expansion, cfg.N_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.N_CHANNELS),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: [B, 1, 28, 28]
        out = F.relu(self.bn1(self.conv1(x)))  # -> [B, 64, 28, 28]
        out = self.layer1(out)  # -> [B, 64, 28, 28]
        out = self.layer2(out)  # -> [B, 128, 14, 14]
        out = self.layer3(out)  # -> [B, 256, 7, 7]
        out = self.final_conv(out)  # -> [B, 32, 7, 7] (匹配 cfg.N_CHANNELS 和 IMG_DIM)
        return out


# =========================================================================
# DFM-FNCN 动态模糊分类层 (保持核心逻辑不变，微调稳定性)
# =========================================================================

class Dynamic_DFM_FNCN(nn.Module):
    def __init__(self, n_channels, p_dim, n_classes, max_rules=cfg.MAX_RULES, phi_th=cfg.PHI_TH):
        super(Dynamic_DFM_FNCN, self).__init__()
        self.n_channels = n_channels
        self.p_dim = p_dim
        self.n_classes = n_classes
        self.max_rules = max_rules
        self.phi_th = phi_th
        self.num_active_rules = 0
        self.pending_rule_data = None

        # 使用 InstanceNorm 而不是 BatchNorm，有时对动态变化的规则层更稳定
        # 或者沿用 BatchNorm 但要注意它需要足够的 batch size 才能统计准确
        self.bn = nn.BatchNorm2d(n_channels)

        self.centers = nn.Parameter(torch.zeros(max_rules, n_channels, p_dim))
        # 初始化宽度时，让它稍微大一点，增加初始覆盖范围，可能有助于冷启动
        self.widths_param = nn.Parameter(torch.ones(max_rules, n_channels) * cfg.INIT_SIGMA)
        self.consequents = nn.Parameter(torch.zeros(max_rules, n_classes))

    def forward(self, x, labels=None, training_phase=False):
        b = x.size(0)
        x = self.bn(x)
        x_flat = x.view(b, self.n_channels, -1)  # [B, N, P]

        # 处理初始无规则状态
        if self.num_active_rules == 0:
            if training_phase and labels is not None:
                self._add_rule_immediately(x_flat[0], labels[0])
            else:
                # 测试时如果没规则，返回均匀分布
                return torch.ones(b, self.n_classes).to(x.device) / self.n_classes

        # 提取活跃参数
        active_centers = self.centers[:self.num_active_rules]
        active_widths_param = self.widths_param[:self.num_active_rules]
        active_consequents = self.consequents[:self.num_active_rules]

        # --- 模糊推理 ---
        x_exp = x_flat.unsqueeze(1)  # [B, 1, N, P]
        c_exp = active_centers.unsqueeze(0)  # [1, R, N, P]

        # 计算余弦相似度 [B, R, N]
        M = F.cosine_similarity(x_exp, c_exp, dim=3)
        d = 1.0 - M

        # 计算宽度和隶属度
        sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
        # 添加一个小的 epsilon 到分母防止除零风险
        mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

        # 计算激发强度 phi [B, R]
        phi = torch.prod(mu, dim=2)

        # --- 在线规则生成检查 ---
        if training_phase and labels is not None and self.num_active_rules < self.max_rules:
            if self.pending_rule_data is None:
                max_phi, _ = phi.max(dim=1)
                # 找到覆盖度最差的样本
                min_phi_val, min_phi_idx = max_phi.min(dim=0)
                if min_phi_val < self.phi_th:
                    # 暂存这个最差样本用于生成新规则
                    self.pending_rule_data = (x_flat[min_phi_idx].detach(), labels[min_phi_idx].detach())

        # --- 去模糊化 ---
        phi_sum = torch.sum(phi, dim=1, keepdim=True) + 1e-9
        phi_norm = phi / phi_sum

        output_logits = torch.matmul(phi_norm, active_consequents)
        return output_logits

    def _add_rule_immediately(self, center_feat, label):
        """仅在 num_rules=0 时调用"""
        with torch.no_grad():
            self._init_rule_at_index(0, center_feat, label)
            self.num_active_rules += 1
            print(f"--> [Init] 初始规则 #1 created for class {label.item()}")

    def commit_pending_rule(self):
        """在 optimizer.step() 后调用"""
        if self.pending_rule_data is not None and self.num_active_rules < self.max_rules:
            with torch.no_grad():
                center, label = self.pending_rule_data
                self._init_rule_at_index(self.num_active_rules, center, label)
                self.num_active_rules += 1
                # print(f"--> [Dynamic] 规则增加至 {self.num_active_rules} (目标类: {label.item()})")
            self.pending_rule_data = None

    def _init_rule_at_index(self, idx, center_feat, label):
        # 1. Center: 用样本特征初始化
        self.centers[idx].copy_(center_feat)
        # 2. Width: 用预设值初始化
        init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1)
        self.widths_param[idx].fill_(init_val)
        # 3. Consequent: 强指向目标类别，其他类别给一个小负值抑制
        # 这样 Softmax 后目标类别的概率会接近 1
        one_hot = torch.ones(self.n_classes, device=self.centers.device) * -1.0
        one_hot[label] = 1.0
        self.consequents[idx].copy_(one_hot)


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        # 使用新的强大的特征提取器
        self.extractor = ResNetFeatureExtractor()
        self.classifier = Dynamic_DFM_FNCN(
            n_channels=cfg.N_CHANNELS,
            p_dim=cfg.P_DIM,
            n_classes=cfg.N_CLASSES
        )

    def forward(self, x, labels=None, training_phase=False):
        features = self.extractor(x)
        logits = self.classifier(features, labels=labels, training_phase=training_phase)
        return logits