import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import config as cfg


# =========================================================================
# 特征提取器 (Encoders)
# =========================================================================

class SimpleCNN(nn.Module):
    """选项 1: 'SIMPLE_CNN'"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, cfg.N_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.N_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PretrainedResNetFeatureExtractor(nn.Module):
    """选项 2: 'RESNET18_PRETRAINED'"""

    def __init__(self):
        super(PretrainedResNetFeatureExtractor, self).__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
        except AttributeError:
            base_model = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = nn.Identity()
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.final_project = nn.Sequential(
            nn.Conv2d(256, cfg.N_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.N_CHANNELS),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_project(x)
        return x


class VGG16FeatureExtractor(nn.Module):
    """[新] 选项 3: 'VGG16_PRETRAINED'"""

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT
            base_model = models.vgg16(weights=weights)
        except AttributeError:
            base_model = models.vgg16(pretrained=True)

        features = list(base_model.features.children())

        # 1. 修改第一层以接受 1 通道输入
        features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # 2. [关键修复] 截断 VGG16。
        # VGG16 结构:
        # ...
        # [9]: MaxPool2d (输出 7x7)
        # [10]-[15]: Conv layers
        # [16]: MaxPool2d (输出 3x3) <-- 错误发生点
        #
        # 我们必须在第 16 层 (MaxPool2d) 之前停止，以获取 7x7 的输出。
        self.features = nn.Sequential(*features[:16])  # 之前是 [:17]

        # 3. 投影层，将 256 通道压缩到 N_CHANNELS (32)
        # (第15层[索引14]的输出通道是 256)
        self.final_project = nn.Sequential(
            nn.Conv2d(256, cfg.N_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.N_CHANNELS),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.final_project(x)
        return x


def get_extractor():
    """工厂函数：根据 config.py 返回所需的特征提取器"""
    if cfg.EXTRACTOR_TYPE == 'RESNET18_PRETRAINED':
        print("使用提取器: PretrainedResNetFeatureExtractor (ResNet18)")
        return PretrainedResNetFeatureExtractor()
    elif cfg.EXTRACTOR_TYPE == 'VGG16_PRETRAINED':
        print("使用提取器: VGG16FeatureExtractor (VGG16)")
        return VGG16FeatureExtractor()
    elif cfg.EXTRACTOR_TYPE == 'SIMPLE_CNN':
        print("使用提取器: SimpleCNN")
        return SimpleCNN()
    else:
        raise ValueError(f"未知的 EXTRACTOR_TYPE: {cfg.EXTRACTOR_TYPE}")


# =========================================================================
# 模型 1: DFM-FNCN (论文复现)
# =========================================================================
class Dynamic_DFM_FNCN(nn.Module):
    def __init__(self, n_channels, p_dim, n_classes, max_rules=cfg.MAX_RULES, phi_th=cfg.PHI_TH):
        super(Dynamic_DFM_FNCN, self).__init__()
        self.n_channels, self.p_dim, self.n_classes = n_channels, p_dim, n_classes
        self.max_rules, self.phi_th = max_rules, phi_th
        self.pending_rule_data = None
        self.register_buffer('num_active_rules', torch.tensor(0, dtype=torch.long))

        self.bn = nn.BatchNorm2d(n_channels)
        self.centers = nn.Parameter(torch.zeros(max_rules, n_channels, p_dim))
        self.widths_param = nn.Parameter(torch.ones(max_rules, n_channels))
        self.consequents = nn.Parameter(torch.zeros(max_rules, n_classes))

    def forward(self, x, labels=None, training_phase=False):
        b = x.size(0)
        x = self.bn(x)
        x_flat = x.view(b, self.n_channels, -1)

        active_rules_count = self.num_active_rules.item()
        if active_rules_count == 0:
            if training_phase and labels is not None:
                self._add_rule_immediately(x_flat[0], labels[0])
                active_rules_count = 1
            else:
                return torch.zeros(b, self.n_classes).to(x.device)

        active_centers = self.centers[:active_rules_count]
        active_widths_param = self.widths_param[:active_rules_count]
        active_consequents = self.consequents[:active_rules_count]

        x_exp, c_exp = x_flat.unsqueeze(1), active_centers.unsqueeze(0)
        M = F.cosine_similarity(x_exp, c_exp, dim=3)
        d = 1.0 - M
        sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
        mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

        # 严格连乘 (64位精度)
        phi_double = torch.prod(mu.to(torch.float64), dim=2)
        phi_sum_double = torch.sum(phi_double, dim=1, keepdim=True) + 1e-9
        phi_norm_double = phi_double / phi_sum_double

        phi = phi_double.to(torch.float32)
        if training_phase and active_rules_count < self.max_rules and self.pending_rule_data is None:
            max_phi, _ = phi.max(dim=1)
            min_phi_val, min_idx = max_phi.min(dim=0)
            if min_phi_val < self.phi_th:
                self.pending_rule_data = (x_flat[min_idx].detach(), labels[min_idx].detach())

        output_logits = torch.matmul(phi_norm_double.to(torch.float32), active_consequents)
        return output_logits

    def _add_rule_immediately(self, center_feat, label):
        with torch.no_grad():
            self._init_rule_at_index(0, center_feat, label)
            self.num_active_rules.add_(1)
            print(f"--> [Init] 初始规则 #1 (Class: {label.item()})")

    def commit_pending_rule(self):
        if self.pending_rule_data is not None and self.num_active_rules.item() < self.max_rules:
            with torch.no_grad():
                center, label = self.pending_rule_data
                self._init_rule_at_index(self.num_active_rules.item(), center, label)
                self.num_active_rules.add_(1)
            self.pending_rule_data = None

    def _init_rule_at_index(self, idx, center_feat, label):
        self.centers[idx].copy_(center_feat)
        init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1) if cfg.INIT_SIGMA > 1e-6 else -5.0
        self.widths_param[idx].fill_(init_val)
        new_consequent = torch.zeros(self.n_classes, device=self.centers.device)
        new_consequent[label] = 2.0
        self.consequents[idx].copy_(new_consequent)


class FullModel(nn.Module):
    """完整模型 (DFM-FNCN)"""

    def __init__(self):
        super(FullModel, self).__init__()
        self.extractor = get_extractor()
        self.classifier = Dynamic_DFM_FNCN(
            n_channels=cfg.N_CHANNELS, p_dim=cfg.P_DIM, n_classes=cfg.N_CLASSES
        )

    def forward(self, x, labels=None, training_phase=False):
        features = self.extractor(x)
        logits = self.classifier(features, labels=labels, training_phase=training_phase)
        return logits


# =========================================================================
# 模型 2: 传统 DCNN (用于对比)
# =========================================================================
class TraditionalCNNModel(nn.Module):
    """传统 DCNN 基线模型"""

    def __init__(self):
        super(TraditionalCNNModel, self).__init__()
        self.extractor = get_extractor()
        flat_features_in = cfg.N_CHANNELS * cfg.P_DIM

        layers = []
        nodes_in = flat_features_in
        for nodes_out in cfg.CNN_CLASSIFIER_NODES:
            layers.append(nn.Linear(nodes_in, nodes_out))
            layers.append(nn.BatchNorm1d(nodes_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.CNN_DROPOUT))
            nodes_in = nodes_out

        layers.append(nn.Linear(nodes_in, cfg.N_CLASSES))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        features = self.extractor(x)
        x = torch.flatten(features, 1)
        logits = self.classifier(x)
        return logits