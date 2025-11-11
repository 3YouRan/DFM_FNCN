import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import config as cfg


# =========================================================================
# 特征提取器 (保持不变)
# =========================================================================
class PretrainedResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(PretrainedResNetFeatureExtractor, self).__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
        except AttributeError:
            print("警告: 您的 torchvision 版本较旧，将使用 pretrained=True 加载。")
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


# =========================================================================
# DFM-FNCN 动态模糊分类层 (修复状态保存)
# =========================================================================
class Dynamic_DFM_FNCN(nn.Module):
    def __init__(self, n_channels, p_dim, n_classes, max_rules=cfg.MAX_RULES, phi_th=cfg.PHI_TH):
        super(Dynamic_DFM_FNCN, self).__init__()
        self.n_channels = n_channels
        self.p_dim = p_dim
        self.n_classes = n_classes
        self.max_rules = max_rules
        self.phi_th = phi_th
        self.pending_rule_data = None

        # [关键修复] 将 num_active_rules 注册为缓冲区
        # 这样它就会被保存在 state_dict 中
        self.register_buffer('num_active_rules', torch.tensor(0, dtype=torch.long))

        self.bn = nn.BatchNorm2d(n_channels)
        self.centers = nn.Parameter(torch.zeros(max_rules, n_channels, p_dim))
        self.widths_param = nn.Parameter(torch.ones(max_rules, n_channels))
        self.consequents = nn.Parameter(torch.zeros(max_rules, n_classes))

    def forward(self, x, labels=None, training_phase=False):
        b = x.size(0)
        x = self.bn(x)
        x_flat = x.view(b, self.n_channels, -1)

        # [关键修复] 使用 .item() 来获取缓冲区的 Python 值
        if self.num_active_rules.item() == 0:
            if training_phase and labels is not None:
                self._add_rule_immediately(x_flat[0], labels[0])
            else:
                # 推理时如果规则为 0 (模型未加载)，返回 0
                return torch.zeros(b, self.n_classes).to(x.device)

        # [关键修复] 使用 .item() 来切片
        active_rules_count = self.num_active_rules.item()
        active_centers = self.centers[:active_rules_count]
        active_widths_param = self.widths_param[:active_rules_count]
        active_consequents = self.consequents[:active_rules_count]

        # --- 模糊推理 (64位精度) ---
        x_exp = x_flat.unsqueeze(1)
        c_exp = active_centers.unsqueeze(0)

        M = F.cosine_similarity(x_exp, c_exp, dim=3)
        d = 1.0 - M
        sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
        mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

        # [Eq. 7] 严格连乘 (在 64 位下)
        mu_double = mu.to(torch.float64)
        phi_double = torch.prod(mu_double, dim=2)

        # --- [Eq. 8] 去模糊化 (在 64 位下) ---
        phi_sum_double = torch.sum(phi_double, dim=1, keepdim=True) + 1e-9
        phi_norm_double = phi_double / phi_sum_double

        # --- 在线规则生成检查 ---
        phi = phi_double.to(torch.float32)
        if training_phase and self.num_active_rules.item() < self.max_rules:
            if self.pending_rule_data is None:
                max_phi, _ = phi.max(dim=1)
                min_phi_val, min_idx = max_phi.min(dim=0)
                if min_phi_val < self.phi_th:
                    self.pending_rule_data = (x_flat[min_idx].detach(), labels[min_idx].detach())

        # --- 最终输出 ---
        phi_norm = phi_norm_double.to(torch.float32)
        output_logits = torch.matmul(phi_norm, active_consequents)
        return output_logits

    def _add_rule_immediately(self, center_feat, label):
        with torch.no_grad():
            self._init_rule_at_index(0, center_feat, label)
            # [关键修复] 使用 .add_() 对缓冲区进行原地加法
            self.num_active_rules.add_(1)
            print(f"--> [Init] 初始规则 #1 (Class: {label.item()})")

    def commit_pending_rule(self):
        if self.pending_rule_data is not None and self.num_active_rules.item() < self.max_rules:
            with torch.no_grad():
                center, label = self.pending_rule_data
                self._init_rule_at_index(self.num_active_rules.item(), center, label)
                # [关键修复] 使用 .add_()
                self.num_active_rules.add_(1)
            self.pending_rule_data = None

    def _init_rule_at_index(self, idx, center_feat, label):
        self.centers[idx].copy_(center_feat)
        init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1)
        self.widths_param[idx].fill_(init_val)
        new_consequent = torch.zeros(self.n_classes, device=self.centers.device)
        new_consequent[label] = 2.0
        self.consequents[idx].copy_(new_consequent)


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.extractor = PretrainedResNetFeatureExtractor()
        self.classifier = Dynamic_DFM_FNCN(
            n_channels=cfg.N_CHANNELS,
            p_dim=cfg.P_DIM,
            n_classes=cfg.N_CLASSES
        )

    def forward(self, x, labels=None, training_phase=False):
        features = self.extractor(x)
        logits = self.classifier(features, labels=labels, training_phase=training_phase)
        return logits