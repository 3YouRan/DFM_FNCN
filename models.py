import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import config as cfg
# [修改] 使用 torch.amp.autocast
from torch.amp.autocast_mode import autocast


# =========================================================================
# 特征提取器 (Encoders)
# =========================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, cfg.N_CHANNELS_OUT, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.N_CHANNELS_OUT), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PretrainedResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(PretrainedResNetFeatureExtractor, self).__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
        except AttributeError:
            base_model = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(cfg.IN_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = nn.Identity()
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.final_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.final_project = nn.Sequential(
            nn.Conv2d(256, cfg.N_CHANNELS_OUT, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.N_CHANNELS_OUT),
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
        x = self.final_pool(x)
        x = self.final_project(x)
        return x


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT
            base_model = models.vgg16(weights=weights)
        except AttributeError:
            base_model = models.vgg16(pretrained=True)

        features = list(base_model.features.children())
        features[0] = nn.Conv2d(cfg.IN_CHANNELS, 64, kernel_size=3, padding=1)
        self.features = nn.Sequential(*features[:16])
        self.final_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.final_project = nn.Sequential(
            nn.Conv2d(256, cfg.N_CHANNELS_OUT, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.N_CHANNELS_OUT),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = self.final_project(x)
        return x


def get_extractor():
    if cfg.EXTRACTOR_TYPE == 'RESNET18_PRETRAINED':
        return PretrainedResNetFeatureExtractor()
    elif cfg.EXTRACTOR_TYPE == 'VGG16_PRETRAINED':
        return VGG16FeatureExtractor()
    elif cfg.EXTRACTOR_TYPE == 'SIMPLE_CNN':
        return SimpleCNN()
    else:
        raise ValueError(f"Unknown extractor: {cfg.EXTRACTOR_TYPE}")


# =========================================================================
# 模型 1: DFM-FNCN (论文复现 + Attention 改进)
# =========================================================================


# =========================================================================
# CBAM 注意力模块
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


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class SEAttention(nn.Module):
    """SE (Squeeze-and-Excitation) 注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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

        # [创新点 1] Attention 参数
        if cfg.USE_ATTENTION:
            if cfg.ATTENTION_TYPE == 'CBAM':
                # CBAM 注意力模块 - 用于特征提取器
                self.cbam_attention = CBAM(n_channels, reduction=cfg.CBAM_REDUCTION, kernel_size=cfg.CBAM_KERNEL_SIZE)
                # 规则级别的注意力权重 (alpha)
                self.alpha = nn.Parameter(torch.zeros(max_rules, n_channels))
            elif cfg.ATTENTION_TYPE == 'SE':
                # SE 注意力模块
                self.se_attention = SEAttention(n_channels, reduction=cfg.CBAM_REDUCTION)
                self.alpha = nn.Parameter(torch.zeros(max_rules, n_channels))
            else:
                # 原始简单注意力
                self.cbam_attention = None
                self.se_attention = None
                self.alpha = nn.Parameter(torch.zeros(max_rules, n_channels))
        else:
            self.register_parameter('alpha', None)
            self.cbam_attention = None
            self.se_attention = None

    def forward(self, x, labels=None, training_phase=False):
        # [修改] 强制使用 float32/float64 避免下溢，并使用 torch.amp.autocast
        with autocast(device_type=cfg.DEVICE.type, enabled=False):
            x = x.to(torch.float32)
            b = x.size(0)
            x = self.bn(x)
            
            # [CBAM 注意力] 在 BN 后应用注意力机制
            if cfg.USE_ATTENTION and cfg.ATTENTION_TYPE == 'CBAM' and self.cbam_attention is not None:
                x = self.cbam_attention(x)
            elif cfg.USE_ATTENTION and cfg.ATTENTION_TYPE == 'SE' and self.se_attention is not None:
                x = self.se_attention(x)
            
            x_flat = x.view(b, self.n_channels, -1)

            active_rules_count = self.num_active_rules.item() # type: ignore
            if active_rules_count == 0:
                if training_phase and labels is not None:
                    self._add_rule_immediately(x_flat[0], labels[0])
                    active_rules_count = 1
                else:
                    return torch.zeros(b, self.n_classes).to(x.device)

            active_centers = self.centers[:active_rules_count]
            active_widths_param = self.widths_param[:active_rules_count]
            active_consequents = self.consequents[:active_rules_count]

            # 计算隶属度 mu
            x_exp, c_exp = x_flat.unsqueeze(1), active_centers.unsqueeze(0)
            M = F.cosine_similarity(x_exp, c_exp, dim=3)
            d = 1.0 - M
            sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
            mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

            # [创新点 1 修正] 聚合逻辑
            if cfg.USE_ATTENTION and self.alpha is not None:
                # 方案: 加权乘积 (Weighted Product)
                # 逻辑: AND (所有重要特征必须匹配)

                # 1. 获取权重 (Softmax 保证和为 1)
                active_alpha = self.alpha[:active_rules_count]
                att_weights = F.softmax(active_alpha, dim=1).unsqueeze(0)  # (1, Rules, Channels)

                # 2. 对数空间计算
                # log(mu^w) = w * log(mu)
                log_mu = torch.log(mu.to(torch.float64) + 1e-9)

                # 3. 加权乘积  (在对数空间)
                # 关键: 乘以 n_channels 以保持量级与原始 Product 一致
                # 如果不乘，结果是几何平均值 (~0.9)，会导致 PHI_TH 失效
                weighted_log_sum = torch.sum(att_weights.to(torch.float64) * log_mu, dim=2) * self.n_channels

                # 4. 转换回线性空间
                phi_double = torch.exp(weighted_log_sum)

            else:
                # 原始连乘 (Product)
                phi_double = torch.prod(mu.to(torch.float64), dim=2)

            # 归一化激发强度
            phi_sum_double = torch.sum(phi_double, dim=1, keepdim=True) + 1e-9
            phi_norm_double = phi_double / phi_sum_double

            # 动态生成规则逻辑
            phi = phi_double.to(torch.float32)
            if training_phase and active_rules_count < self.max_rules and self.pending_rule_data is None:
                max_phi, _ = phi.max(dim=1)
                min_phi_val, min_idx = max_phi.min(dim=0)
                if min_phi_val < self.phi_th:
                    self.pending_rule_data = (x_flat[min_idx].detach(), labels[min_idx].detach()) # type: ignore

            output_logits = torch.matmul(phi_norm_double.to(torch.float32), active_consequents)

        return output_logits

    def get_rule_activations(self, x):
        """[创新点 2] 获取规则的激活强度 (Phi)，用于基于激活的修剪"""
        with torch.no_grad():
            x = x.to(torch.float32)
            b = x.size(0)
            x = self.bn(x)
            
            # [CBAM 注意力] 在 BN 后应用注意力机制
            if cfg.USE_ATTENTION and cfg.ATTENTION_TYPE == 'CBAM' and self.cbam_attention is not None:
                x = self.cbam_attention(x)
            elif cfg.USE_ATTENTION and cfg.ATTENTION_TYPE == 'SE' and self.se_attention is not None:
                x = self.se_attention(x)
            
            x_flat = x.view(b, self.n_channels, -1)

            active_rules_count = self.num_active_rules.item() # type: ignore
            if active_rules_count == 0:
                return None

            active_centers = self.centers[:active_rules_count]
            active_widths_param = self.widths_param[:active_rules_count]

            x_exp, c_exp = x_flat.unsqueeze(1), active_centers.unsqueeze(0)
            M = F.cosine_similarity(x_exp, c_exp, dim=3)
            d = 1.0 - M
            sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6
            mu = torch.exp(-torch.pow(d, 2) / (torch.pow(sigma, 2) + 1e-8))

            if cfg.USE_ATTENTION and self.alpha is not None:
                active_alpha = self.alpha[:active_rules_count]
                att_weights = F.softmax(active_alpha, dim=1).unsqueeze(0)
                log_mu = torch.log(mu.to(torch.float64) + 1e-9)
                weighted_log_sum = torch.sum(att_weights.to(torch.float64) * log_mu, dim=2) * self.n_channels
                phi_double = torch.exp(weighted_log_sum)
            else:
                phi_double = torch.prod(mu.to(torch.float64), dim=2)

            # 返回归一化的激活强度
            phi_sum_double = torch.sum(phi_double, dim=1, keepdim=True) + 1e-9
            phi_norm_double = phi_double / phi_sum_double
            return phi_norm_double.to(torch.float32)

    def prune_rules(self, keep_indices):
        """[创新点 2] 物理修剪规则，仅保留 keep_indices 中的规则"""
        if len(keep_indices) == 0:
            print("警告: 尝试修剪所有规则，操作已取消。")
            return

        with torch.no_grad():
            keep_indices = torch.tensor(keep_indices, device=self.centers.device, dtype=torch.long)
            new_count = len(keep_indices)

            # 1. 提取要保留的参数
            kept_centers = self.centers[keep_indices].clone()
            kept_widths = self.widths_param[keep_indices].clone()
            kept_consequents = self.consequents[keep_indices].clone()

            # 2. 重置所有参数
            self.centers.zero_()
            self.widths_param.fill_(0)
            self.consequents.zero_()

            # 3. 填回保留的参数到前部
            self.centers[:new_count] = kept_centers
            self.widths_param[:new_count] = kept_widths
            self.consequents[:new_count] = kept_consequents

            # 4. 处理 Attention 参数
            if cfg.USE_ATTENTION and self.alpha is not None:
                kept_alpha = self.alpha[keep_indices].clone()
                self.alpha.zero_()
                self.alpha[:new_count] = kept_alpha

            # 5. 更新计数
            old_count = self.num_active_rules.item() # type: ignore
            self.num_active_rules.fill_(new_count) # type: ignore
            print(f"--> [Pruning] 规则数从 {old_count} 减少到 {new_count}。")

    def _add_rule_immediately(self, center_feat, label):
        with torch.no_grad():
            self._init_rule_at_index(0, center_feat, label)
            self.num_active_rules.add_(1) # type: ignore
            print(f"--> [Init] 初始规则 #1 (Class: {label.item()})")

    def commit_pending_rule(self):
        if self.pending_rule_data is not None and self.num_active_rules.item() < self.max_rules: # type: ignore
            with torch.no_grad():
                center, label = self.pending_rule_data
                self._init_rule_at_index(self.num_active_rules.item(), center, label) # type: ignore
                self.num_active_rules.add_(1) # type: ignore
            self.pending_rule_data = None

    def _init_rule_at_index(self, idx, center_feat, label):
        self.centers[idx].copy_(center_feat)
        init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1) if cfg.INIT_SIGMA > 1e-6 else -5.0
        self.widths_param[idx].fill_(init_val)
        new_consequent = torch.zeros(self.n_classes, device=self.centers.device)
        new_consequent[label] = 2.0
        self.consequents[idx].copy_(new_consequent)

    def init_rules_from_cluster_centers(self, centers_tensor, majority_classes):
        """
        [创新点 3] 批量初始化规则
        centers_tensor: (K, C, P)
        majority_classes: list of int, length K
        """
        num_init = centers_tensor.size(0)
        if num_init > self.max_rules:
            print(f"警告: 聚类数 {num_init} 大于最大规则数 {self.max_rules}，将被截断。")
            num_init = self.max_rules
            centers_tensor = centers_tensor[:num_init]
            majority_classes = majority_classes[:num_init]

        with torch.no_grad():
            # 1. 设置中心
            self.centers[:num_init].copy_(centers_tensor)

            # 2. 设置宽度 (初始化为默认值)
            init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1) if cfg.INIT_SIGMA > 1e-6 else -5.0
            self.widths_param[:num_init].fill_(init_val)

            # 3. 设置后件
            self.consequents.zero_()  # Reset
            for i, label in enumerate(majority_classes):
                self.consequents[i, label] = 2.0

            # 4. 更新激活规则数
            self.num_active_rules.fill_(num_init) # type: ignore
            print(f"--> [Batch Init] 已批量初始化 {num_init} 条规则。")

    def merge_similar_rules(self, test_loader=None):
        """[创新点 6] 动态融合相似规则"""
        if not cfg.USE_RULE_MERGING:
            return 0, []

        active_rules_count = self.num_active_rules.item() # type: ignore
        if active_rules_count <= 1:
            return 0, []

        print(f"\n>>> 正在执行规则融合 (Method: {cfg.MERGING_METHOD}, Th: {cfg.MERGING_THRESHOLD})...")

        with torch.no_grad():
            # 获取当前激活的规则参数
            active_centers = self.centers[:active_rules_count]  # (R, C, P)
            active_consequents = self.consequents[:active_rules_count]  # (R, Classes)
            active_widths = self.widths_param[:active_rules_count]  # (R, C)

            if cfg.USE_ATTENTION and self.alpha is not None:
                active_alpha = self.alpha[:active_rules_count]  # (R, C)
            else:
                active_alpha = None

            # 计算规则相似度矩阵
            similarity_matrix = self._compute_rule_similarity(active_centers, test_loader)

            # 查找需要融合的规则对
            merge_pairs = self._find_merge_pairs(similarity_matrix, active_consequents)

            if not merge_pairs:
                print("没有找到需要融合的规则对。")
                return 0, []

            # 执行融合
            merged_rules_info = self._perform_merging(
                merge_pairs, active_centers, active_widths,
                active_consequents, active_alpha
            )

            # 更新模型参数
            self._update_parameters_after_merging(merged_rules_info)

            # 更新激活规则计数
            new_count = active_rules_count - len(merge_pairs)
            old_count = self.num_active_rules.item() # type: ignore
            self.num_active_rules.fill_(new_count) # type: ignore

            print(f"--> [Merging] 规则数从 {old_count} 减少到 {new_count}，融合了 {len(merge_pairs)} 对规则。")

            return len(merge_pairs), merge_pairs

    def _compute_rule_similarity(self, centers, test_loader=None):
        """计算规则相似度矩阵"""
        active_rules_count = centers.size(0)

        if cfg.MERGING_METHOD == 'SIMILARITY':
            # 基于规则中心的余弦相似度
            # 将中心展平: (R, C, P) -> (R, C*P)
            centers_flat = centers.view(active_rules_count, -1)
            # 计算余弦相似度矩阵
            similarity_matrix = F.cosine_similarity(
                centers_flat.unsqueeze(1),  # (R, 1, D)
                centers_flat.unsqueeze(0),  # (1, R, D)
                dim=2
            )

        elif cfg.MERGING_METHOD == 'ACTIVATION_CORRELATION' and test_loader is not None:
            # 基于规则激活的相关性
            # 收集规则在测试集上的激活模式
            activation_patterns = []

            for data_tuple in test_loader:
                data = data_tuple[0].to(cfg.DEVICE)
                features = self.bn(data.view(data.size(0), self.n_channels, -1))

                # 计算激活强度
                x_exp = features.unsqueeze(1)  # (B, 1, C, P)
                c_exp = centers.unsqueeze(0)   # (1, R, C, P)
                M = F.cosine_similarity(x_exp, c_exp, dim=3)
                d = 1.0 - M
                sigma = F.softplus(self.widths_param[:active_rules_count]).unsqueeze(0) + 1e-6
                mu = torch.exp(-torch.pow(d, 2) / torch.pow(sigma, 2))

                if cfg.USE_ATTENTION and self.alpha is not None:
                    active_alpha = self.alpha[:active_rules_count]
                    att_weights = F.softmax(active_alpha, dim=1).unsqueeze(0)
                    log_mu = torch.log(mu.to(torch.float64) + 1e-9)
                    weighted_log_sum = torch.sum(att_weights.to(torch.float64) * log_mu, dim=2) * self.n_channels
                    phi = torch.exp(weighted_log_sum).to(torch.float32)
                else:
                    phi = torch.prod(mu, dim=2)

                activation_patterns.append(phi.cpu())

                if len(activation_patterns) * data.size(0) > 1000:  # 限制样本数
                    break

            if activation_patterns:
                all_activations = torch.cat(activation_patterns, dim=0)  # (N, R)
                # 计算皮尔逊相关系数
                similarity_matrix = torch.corrcoef(all_activations.T)
            else:
                # 回退到余弦相似度
                centers_flat = centers.view(active_rules_count, -1)
                similarity_matrix = F.cosine_similarity(
                    centers_flat.unsqueeze(1),
                    centers_flat.unsqueeze(0),
                    dim=2
                )
        else:
            # 默认使用余弦相似度
            centers_flat = centers.view(active_rules_count, -1)
            similarity_matrix = F.cosine_similarity(
                centers_flat.unsqueeze(1),
                centers_flat.unsqueeze(0),
                dim=2
            )

        return similarity_matrix

    def _find_merge_pairs(self, similarity_matrix, consequents):
        """查找需要融合的规则对"""
        active_rules_count = similarity_matrix.size(0)
        merge_pairs = []
        merged_indices = set()

        # 计算每条规则的置信度（后件最大概率）
        consequents_prob = F.softmax(consequents, dim=1)
        rule_confidences = torch.max(consequents_prob, dim=1)[0]

        for i in range(active_rules_count):
            if i in merged_indices:
                continue

            for j in range(i + 1, active_rules_count):
                if j in merged_indices:
                    continue

                # 检查相似度是否超过阈值
                if similarity_matrix[i, j] >= cfg.MERGING_THRESHOLD:
                    # 检查后件是否相似（预测同一类别）
                    pred_i = torch.argmax(consequents_prob[i])
                    pred_j = torch.argmax(consequents_prob[j])

                    if pred_i == pred_j:
                        merge_pairs.append((i, j))
                        merged_indices.add(i)
                        merged_indices.add(j)
                        break  # 每条规则只融合一次

        return merge_pairs

    def _perform_merging(self, merge_pairs, centers, widths, consequents, alpha=None):
        """执行规则融合"""
        merged_rules_info = []

        for i, j in merge_pairs:
            # 获取两条规则的参数
            center_i, center_j = centers[i], centers[j]
            width_i, width_j = widths[i], widths[j]
            consequent_i, consequent_j = consequents[i], consequents[j]

            # 计算规则置信度（用于加权）
            consequent_prob_i = F.softmax(consequent_i, dim=0)
            consequent_prob_j = F.softmax(consequent_j, dim=0)
            conf_i = torch.max(consequent_prob_i)
            conf_j = torch.max(consequent_prob_j)

            if cfg.MERGING_STRATEGY == 'WEIGHTED_AVERAGE':
                # 加权平均融合
                total_conf = conf_i + conf_j + 1e-8
                weight_i = conf_i / total_conf
                weight_j = conf_j / total_conf

                merged_center = weight_i * center_i + weight_j * center_j
                merged_width = weight_i * width_i + weight_j * width_j
                merged_consequent = weight_i * consequent_i + weight_j * consequent_j

            elif cfg.MERGING_STRATEGY == 'DOMINANT_RULE':
                # 保留置信度更高的规则
                if conf_i >= conf_j:
                    merged_center = center_i
                    merged_width = width_i
                    merged_consequent = consequent_i
                else:
                    merged_center = center_j
                    merged_width = width_j
                    merged_consequent = consequent_j

            # 处理注意力权重
            merged_alpha = None
            if alpha is not None:
                alpha_i, alpha_j = alpha[i], alpha[j]
                if cfg.MERGING_STRATEGY == 'WEIGHTED_AVERAGE':
                    merged_alpha = weight_i * alpha_i + weight_j * alpha_j # type: ignore
                elif cfg.MERGING_STRATEGY == 'DOMINANT_RULE':
                    if conf_i >= conf_j:
                        merged_alpha = alpha_i
                    else:
                        merged_alpha = alpha_j

            merged_rules_info.append({
                'center': merged_center, # type: ignore
                'width': merged_width, # type: ignore
                'consequent': merged_consequent, # type: ignore
                'alpha': merged_alpha,
                'original_indices': (i, j),
                'confidences': (conf_i.item(), conf_j.item())
            })

        return merged_rules_info

    def _update_parameters_after_merging(self, merged_rules_info):
        """融合后更新模型参数"""
        if not merged_rules_info:
            return

        active_rules_count = self.num_active_rules.item() # type: ignore
        merge_pairs = [info['original_indices'] for info in merged_rules_info]

        # 找出需要保留的规则索引（未被融合的规则）
        all_indices = set(range(active_rules_count)) # type: ignore
        merged_indices = set()
        for i, j in merge_pairs:
            merged_indices.add(i)
            merged_indices.add(j)
        keep_indices = sorted(list(all_indices - merged_indices))

        # 创建新的参数数组
        new_count = len(keep_indices) + len(merged_rules_info)
        new_centers = torch.zeros_like(self.centers[:new_count])
        new_widths = torch.zeros_like(self.widths_param[:new_count])
        new_consequents = torch.zeros_like(self.consequents[:new_count])

        if cfg.USE_ATTENTION and self.alpha is not None:
            new_alpha = torch.zeros_like(self.alpha[:new_count])
        else:
            new_alpha = None

        # 1. 填充未被融合的规则
        for idx, rule_idx in enumerate(keep_indices):
            new_centers[idx] = self.centers[rule_idx].clone()
            new_widths[idx] = self.widths_param[rule_idx].clone()
            new_consequents[idx] = self.consequents[rule_idx].clone()
            if new_alpha is not None:
                new_alpha[idx] = self.alpha[rule_idx].clone()

        # 2. 填充融合后的规则
        offset = len(keep_indices)
        for idx, rule_info in enumerate(merged_rules_info):
            new_centers[offset + idx] = rule_info['center']
            new_widths[offset + idx] = rule_info['width']
            new_consequents[offset + idx] = rule_info['consequent']
            if new_alpha is not None and rule_info['alpha'] is not None:
                new_alpha[offset + idx] = rule_info['alpha']

        # 3. 更新模型参数
        with torch.no_grad():
            # 清空原始参数
            self.centers.zero_()
            self.widths_param.fill_(0)
            self.consequents.zero_()

            # 填充新参数
            self.centers[:new_count] = new_centers
            self.widths_param[:new_count] = new_widths
            self.consequents[:new_count] = new_consequents

            if new_alpha is not None:
                self.alpha.zero_()
                self.alpha[:new_count] = new_alpha


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.extractor = get_extractor()
        self.classifier = Dynamic_DFM_FNCN(
            n_channels=cfg.N_CHANNELS_OUT, p_dim=cfg.P_DIM, n_classes=cfg.N_CLASSES
        )

    def forward(self, x, labels=None, training_phase=False):
        features = self.extractor(x)
        logits = self.classifier(features, labels=labels, training_phase=training_phase)
        return logits


class TraditionalCNNModel(nn.Module):
    def __init__(self):
        super(TraditionalCNNModel, self).__init__()
        self.extractor = get_extractor()
        flat_features_in = cfg.N_CHANNELS_OUT * cfg.P_DIM
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


# =========================================================================
# PlantNet 对比算法 (复现论文中的 Fuzzy CNN 模型)
# =========================================================================

class GaussianFuzzyLayer(nn.Module):
    """
    实现论文 Table 2 中提到的 "Fuzzy Layer Type: Gaussian"。
    这是一个具有可学习参数（均值和标准差）的层，模拟 ANFIS 中的模糊化过程。
    """
    def __init__(self, in_channels, num_membership_functions=2):
        super(GaussianFuzzyLayer, self).__init__()
        self.in_channels = in_channels
        self.k = num_membership_functions

        # 可学习的参数：均值 (mu) 和 标准差 (sigma)
        # 初始化 mu 为 0 附近，sigma 为 1 附近
        self.mu = nn.Parameter(torch.randn(in_channels, self.k) * 0.1)
        self.sigma = nn.Parameter(torch.ones(in_channels, self.k))

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        # 我们希望在通道维度上应用模糊隶属函数

        # 将输入扩展以匹配隶属度函数的数量
        # x_expanded: [B, C, 1, H, W]
        x_expanded = x.unsqueeze(2)

        # mu, sigma: [C, K] -> [1, C, K, 1, 1]
        mu_expanded = self.mu.view(1, self.in_channels, self.k, 1, 1)
        sigma_expanded = self.sigma.view(1, self.in_channels, self.k, 1, 1)

        # 高斯隶属函数公式: exp(- (x - mu)^2 / (2 * sigma^2))
        # 加上一个小的 epsilon 防止除零
        membership = torch.exp(-
            torch.pow(x_expanded - mu_expanded, 2) / 
            (2 * torch.pow(sigma_expanded, 2) + 1e-5)
        )

        # [B, C, K, H, W] -> [B, C * K, H, W]
        # 将模糊特征合并回通道维度
        B, C, K, H, W = membership.shape
        out = membership.view(B, C * K, H, W)

        return out


class PlantNetANFIS(nn.Module):
    """
    复现论文中的 ANFIS Fuzzy CNN 模型结构作为对比算法。
    参考用户提供的代码结构和论文 Table 2:
    - Input: (224, 224, 3) 或适配项目配置的尺寸
    - Conv Layers: 32, 64, 128 filters
    - Kernel: 3x3
    - Activation: ReLU
    - Pooling: MaxPool 2x2
    - Fuzzy Layer: Gaussian
    """
    def __init__(self, num_classes=None, use_fuzzy_layer=True, in_channels=3):
        super(PlantNetANFIS, self).__init__()
        
        # 使用传入的 num_classes 或从配置中获取
        num_classes = num_classes if num_classes is not None else cfg.N_CLASSES
        self.use_fuzzy_layer = use_fuzzy_layer
        
        # --- Block 1 ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Fuzzy Block ---
        # 论文指出使用了 Fuzzy Layer。这里我们将卷积特征输入到模糊层。
        self.num_mf = 2  # 每个通道的隶属函数数量
        if self.use_fuzzy_layer:
            self.fuzzy_layer = GaussianFuzzyLayer(
                in_channels=128, 
                num_membership_functions=self.num_mf
            )
        else:
            self.fuzzy_layer = None
        
        # --- Fully Connected ---
        # 使用动态计算，先占位，初始化时通过 dummy input 计算实际维度
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Identity()  # 占位，初始化后替换
        self.fc2 = nn.Linear(512, num_classes)
        
        # 通过 dummy input 计算展平后的维度
        self._init_fc_layers()
    
    def _init_fc_layers(self):
        """通过 dummy input 动态计算全连接层的输入维度"""
        with torch.no_grad():
            # 使用配置中的 TARGET_SIZE
            dummy_input = torch.randn(1, self.conv1.in_channels, *cfg.TARGET_SIZE)
            x = self._forward_features(dummy_input)
            flat_dim = x.view(1, -1).shape[1]
            
            # 重新创建全连接层
            self.fc1 = nn.Linear(flat_dim, 512)
            print(f"PlantNetANFIS: 动态计算的展平维度 = {flat_dim} (输入尺寸: {cfg.TARGET_SIZE})")
    
    def _forward_features(self, x):
        """提取特征的辅助方法"""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        if self.use_fuzzy_layer and self.fuzzy_layer is not None:
            x = self.fuzzy_layer(x)
        
        return x

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        
        # 提取特征
        x = self._forward_features(x)

        # Flatten
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 论文使用的是 Categorical Crossentropy，所以这里输出 logits
        return x


class PlantNetSimple(nn.Module):
    """
    PlantNet 简化版对比算法 (不含模糊层)
    标准的 CNN 架构，用于与 PlantNetANFIS 进行性能对比
    """
    def __init__(self, num_classes=None, in_channels=3):
        super(PlantNetSimple, self).__init__()
        
        num_classes = num_classes if num_classes is not None else cfg.N_CLASSES
        
        # --- Block 1 ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 4 ---
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Fully Connected ---
        # 使用动态计算，先占位，初始化时通过 dummy input 计算实际维度
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Identity()  # 占位，初始化后替换
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # 通过 dummy input 计算展平后的维度
        self._init_fc_layers()
    
    def _init_fc_layers(self):
        """通过 dummy input 动态计算全连接层的输入维度"""
        with torch.no_grad():
            # 使用配置中的 TARGET_SIZE
            dummy_input = torch.randn(1, self.conv1.in_channels, *cfg.TARGET_SIZE)
            x = self._forward_features(dummy_input)
            flat_dim = x.view(1, -1).shape[1]
            
            # 重新创建全连接层
            self.fc1 = nn.Linear(flat_dim, 512)
            print(f"PlantNetSimple: 动态计算的展平维度 = {flat_dim} (输入尺寸: {cfg.TARGET_SIZE})")
    
    def _forward_features(self, x):
        """提取特征的辅助方法"""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def get_plantnet_model(model_type='ANFIS', num_classes=None, in_channels=3):
    """
    获取 PlantNet 模型实例
    
    Args:
        model_type: 'ANFIS' - 带模糊层的ANFIS版本
                    'SIMPLE' - 简化版CNN
        num_classes: 分类数量
        in_channels: 输入通道数
    
    Returns:
        PlantNet 模型实例
    """
    if model_type == 'ANFIS':
        return PlantNetANFIS(num_classes=num_classes, use_fuzzy_layer=True, in_channels=in_channels)
    elif model_type == 'SIMPLE':
        return PlantNetSimple(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown PlantNet model type: {model_type}")


# =========================================================================
# OsteoNet 对比算法 (复现 Abed et al., 2025 论文中的 Fuzzy CNN 模型)
# =========================================================================

class FuzzyContrastEnhancement(nn.Module):
    """
    [顺序架构第一步] 模糊逻辑预处理层 (Fuzzy Preprocessing Layer)
    
    复现论文中的 "Fuzzy Logic Preprocessing" 阶段。
    主要通过模糊集合理论增强图像的对比度，突出细节。
    
    算法流程:
    1. 图像归一化 (Normalization)
    2. 模糊化 (Fuzzification): 将像素值映射为隶属度。
    3. 强化算子 (Intensification): 使用 INT 算子拉伸对比度。
    4. 去模糊化 (Defuzzification): 映射回像素空间。
    """
    def __init__(self, crossover_point=0.5):
        super(FuzzyContrastEnhancement, self).__init__()
        self.crossover_point = crossover_point  # 模糊过渡点，通常为0.5

    def intensification_operator(self, mu):
        """
        模糊强化算子 (Intensification Operator).
        逻辑: 
        - 如果隶属度 < 0.5 (暗部)，则使其更暗。
        - 如果隶属度 > 0.5 (亮部)，则使其更亮。
        公式:
        mu_new = 2 * mu^2             if 0 <= mu <= 0.5
        mu_new = 1 - 2 * (1 - mu)^2   if 0.5 < mu <= 1
        """
        # 使用 torch.where 实现分段函数，保持 GPU 加速能力
        mu_new = torch.where(
            mu <= 0.5,
            2 * torch.pow(mu, 2),
            1 - 2 * torch.pow(1 - mu, 2)
        )
        return mu_new

    def forward(self, img_tensor):
        """
        Args:
            img_tensor: 输入图像张量 [Batch, Channels, Height, Width], 值域通常在 [0, 1] 或 [0, 255]
        """
        # 1. 确保输入在 [0, 1] 范围
        if img_tensor.max() > 1.0:
            x = img_tensor / 255.0
        else:
            x = img_tensor

        # 2. Fuzzification (模糊化)
        # 对于灰度图像，像素亮度即代表"明亮程度"的隶属度
        # 对于彩色图像，取 RGB 平均值作为亮度
        if x.shape[1] == 3:
            # 彩色图像：转换为灰度亮度
            mu = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            # 单通道图像：直接使用
            mu = x

        # 3. Fuzzy Intensification (模糊强化)
        mu_enhanced = self.intensification_operator(mu)
        
        # 4. Defuzzification (去模糊化)
        # 将增强后的亮度映射回所有通道
        if x.shape[1] == 3:
            out = torch.cat([mu_enhanced, mu_enhanced, mu_enhanced], dim=1)
        else:
            out = mu_enhanced

        return out


class OsteoNet(nn.Module):
    """
    [顺序架构第二步] 深度卷积网络后端 (Deep CNN Back-end)
    
    复现论文 Abed et al., 2025 中的 OsteoNet 模型。
    架构: Fuzzy Preprocessing + CNN Backbone
    
    支持的骨干网络: 'resnet18', 'resnet50', 'mobilenet_v2', 'alexnet'
    """
    def __init__(self, model_name='resnet18', num_classes=None, pretrained=True):
        super(OsteoNet, self).__init__()
        
        # 使用传入的 num_classes 或从配置中获取
        num_classes = num_classes if num_classes is not None else cfg.N_CLASSES
        self.model_name = model_name.lower()
        
        # 1. 初始化预处理层 (The Sequential Fuzzy Front-end)
        self.fuzzy_preprocess = FuzzyContrastEnhancement()

        # 2. 加载预训练骨干网络 (The Deep Back-end)
        if 'resnet' in self.model_name:
            if self.model_name == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            elif self.model_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            else:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            
            # 修改首层以适应单通道输入
            if cfg.IN_CHANNELS == 1:
                original_conv = self.backbone.conv1
                self.backbone.conv1 = nn.Conv2d(
                    1, 64, kernel_size=original_conv.kernel_size, 
                    stride=original_conv.stride, padding=original_conv.padding, bias=False
                )
            elif cfg.IN_CHANNELS != 3:
                self.backbone.conv1 = nn.Conv2d(
                    cfg.IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif 'mobilenet' in self.model_name:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            
            # 修改首层以适应单通道输入
            if cfg.IN_CHANNELS == 1:
                # MobileNetV2 的第一个层是 InvertedResidual，我们需要修改它的第一个卷积层
                # 获取原始层的参数
                orig_conv = self.backbone.features[0][0]  # type: ignore
                new_first_conv = nn.Conv2d(
                    1, 32, kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride, padding=orig_conv.padding, bias=False
                )
                # 直接替换 (PyTorch 支持这种动态替换)
                self.backbone.features[0][0] = new_first_conv  # type: ignore
            elif cfg.IN_CHANNELS != 3:
                orig_conv = self.backbone.features[0][0]  # type: ignore
                new_first_conv = nn.Conv2d(
                    cfg.IN_CHANNELS, 32, kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride, padding=orig_conv.padding, bias=False
                )
                self.backbone.features[0][0] = new_first_conv  # type: ignore
            
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif 'alexnet' in self.model_name:
            weights = models.AlexNet_Weights.DEFAULT if pretrained else None
            self.backbone = models.alexnet(weights=weights)
            
            # 修改首层以适应单通道输入
            if cfg.IN_CHANNELS != 3:
                first_conv_config = {
                    'in_channels': cfg.IN_CHANNELS,
                    'out_channels': 64,
                    'kernel_size': 11,
                    'stride': 4,
                    'padding': 2  # 添加 padding 确保输出尺寸
                }
                self.backbone.features[0] = nn.Conv2d(**first_conv_config)
            
            # 修改最后一个 MaxPool2d 层，添加 padding 以适应小尺寸输入
            self.backbone.features[2] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # 使用 AdaptiveAvgPool2d(1, 1) 确保任何输入尺寸都能产生 1x1 输出
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 获取原始 classifier 的输入特征数
            with torch.no_grad():
                dummy = torch.randn(1, cfg.IN_CHANNELS if cfg.IN_CHANNELS != 3 else 3, 
                                   cfg.TARGET_SIZE[0], cfg.TARGET_SIZE[1])
                # 先经过特征提取
                x = self.backbone.features(dummy)
                # 经过 avgpool
                x = self.backbone.avgpool(x)
                in_features = x.shape[1]
            
            # 重建 classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            
        else:
            raise ValueError(f"Model {model_name} not supported. Choose from: resnet18, resnet50, mobilenet_v2, alexnet")

    def forward(self, x):
        """
        Args:
            x: 输入图像张量 [Batch, Channels, Height, Width]
        
        Returns:
            logits: 分类 logits
            enhanced_img: 模糊增强后的图像 (用于可视化)
        """
        # --- Stage 1: Fuzzy Preprocessing ---
        # 图像先经过模糊逻辑层处理，增强特征
        x_enhanced = self.fuzzy_preprocess(x)
        
        # --- Stage 2: Deep Learning Classification ---
        # 增强后的图像输入 CNN
        logits = self.backbone(x_enhanced)
        
        return logits, x_enhanced


class OsteoNetSimple(nn.Module):
    """
    OsteoNet 简化版对比算法
    
    使用轻量级的自定义 CNN 作为骨干网络，适合小规模数据集和快速实验。
    保持 Fuzzy Preprocessing 层，移除预训练模型。
    """
    def __init__(self, num_classes=None, in_channels=3):
        super(OsteoNetSimple, self).__init__()
        
        num_classes = num_classes if num_classes is not None else cfg.N_CLASSES
        
        # Fuzzy Preprocessing Layer
        self.fuzzy_preprocess = FuzzyContrastEnhancement()
        
        # Custom CNN Backbone (轻量级)
        # 输入尺寸: 28x28 -> 经过 3 个 MaxPool 后: 3x3
        self.backbone = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Classifier
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像张量 [Batch, Channels, Height, Width]
        
        Returns:
            logits: 分类 logits
            enhanced_img: 模糊增强后的图像 (用于可视化)
        """
        # Stage 1: Fuzzy Preprocessing
        x_enhanced = self.fuzzy_preprocess(x)
        
        # Stage 2: Classification
        logits = self.backbone(x_enhanced)
        
        return logits, x_enhanced


def get_osteonet_model(model_type='resnet18', num_classes=None, pretrained=True):
    """
    获取 OsteoNet 模型实例
    
    Args:
        model_type: 'resnet18' - ResNet18 预训练骨干
                    'resnet50' - ResNet50 预训练骨干
                    'mobilenet_v2' - MobileNetV2 预训练骨干
                    'alexnet' - AlexNet 预训练骨干
                    'simple' - 自定义轻量级 CNN 骨干
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    
    Returns:
        OsteoNet 模型实例
    """
    if model_type == 'simple':
        return OsteoNetSimple(num_classes=num_classes, in_channels=cfg.IN_CHANNELS)
    else:
        return OsteoNet(model_name=model_type, num_classes=num_classes, pretrained=pretrained)


# =========================================================================
# HP-FCNN 对比算法 (复现 Iqbal et al., 2024 IEEE Trans. Fuzzy Systems)
# =========================================================================

class FuzzyLayer(nn.Module):
    """
    可学习的模糊层 (Learnable Fuzzy Layer)
    作用: 将输入特征映射到模糊隶属度空间，捕捉数据的不确定性。
    实现: 采用高斯隶属函数 (Gaussian Membership Function)。
    """
    def __init__(self, in_channels, num_fuzzy_sets):
        super(FuzzyLayer, self).__init__()
        self.num_fuzzy_sets = num_fuzzy_sets
        self.in_channels = in_channels

        # 初始化高斯函数的中心 (Centers) 和 宽度 (Sigmas)
        # 形状: (1, in_channels, num_fuzzy_sets, 1, 1) 以便广播到图像空间
        self.centers = nn.Parameter(torch.rand(1, in_channels, num_fuzzy_sets, 1, 1))
        self.sigmas = nn.Parameter(torch.ones(1, in_channels, num_fuzzy_sets, 1, 1))

    def forward(self, x):
        """
        x: (Batch, C, H, W)
        Return: (Batch, C * Num_Fuzzy_Sets, H, W)
        """
        B, C, H, W = x.shape
        
        # 扩展 x 以匹配模糊集的维度: (B, C, 1, H, W)
        x_expanded = x.unsqueeze(2)
        
        # 计算高斯隶属度: exp(- (x - c)^2 / (2 * sigma^2))
        # 结果形状: (B, C, Num_Rules, H, W)
        numerator = (x_expanded - self.centers) ** 2
        denominator = 2 * (self.sigmas ** 2) + 1e-8 # 加极小值防止除零
        membership = torch.exp(-numerator / denominator)
        
        # 将模糊集维度合并到通道维度，以便后续卷积处理
        # (B, C * Num_Rules, H, W)
        membership = membership.view(B, C * self.num_fuzzy_sets, H, W)
        
        return membership


class HP_FCNN(nn.Module):
    """
    Hybrid Parallel Fuzzy CNN (HP-FCNN)
    复现对象: Iqbal et al. (2024) IEEE Trans. Fuzzy Systems
    架构特点:
      - Branch A: Deep Crisp CNN (提取纹理、边缘等清晰特征)
      - Branch B: Parallel Fuzzy Stream (提取模糊特征，处理边界不确定性)
      - Fusion: 特征级拼接融合 (Concatenation)
    """
    def __init__(self, num_classes=None, in_channels=3):
        super(HP_FCNN, self).__init__()
        
        # 使用传入的 num_classes 或从配置中获取
        num_classes = num_classes if num_classes is not None else cfg.N_CLASSES
        
        # ---------------------------
        # 分支 A: 清晰卷积流 (Crisp CNN Stream)
        # ---------------------------
        self.crisp_stream = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # /2
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # /4
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # /8
        )
        
        # ---------------------------
        # 分支 B: 模糊流 (Fuzzy Stream)
        # ---------------------------
        # 这里的逻辑是：先通过模糊层提取隶属度，再通过卷积层聚合模糊信息
        self.fuzzy_in_channels = 32 # 假设我们在第一层卷积后分叉
        self.pre_fuzzy_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.fuzzy_in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fuzzy_sets = 4 # 每个通道定义4个模糊语义 (如: Low, Med-Low, Med-High, High)
        self.fuzzy_layer = FuzzyLayer(self.fuzzy_in_channels, self.fuzzy_sets)
        
        # 模糊特征聚合层 (将膨胀的模糊通道压缩回来)
        self.fuzzy_stream_conv = nn.Sequential(
            # 输入通道变大了 (Channel * Fuzzy_Sets)
            nn.Conv2d(self.fuzzy_in_channels * self.fuzzy_sets, 64, kernel_size=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # /2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # /4
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # /8
        )

        # ---------------------------
        # 融合与分类 (Fusion & Classifier)
        # ---------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 融合后的维度 = Crisp输出通道(128) + Fuzzy输出通道(128)
        self.fusion_dim = 128 + 128 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # --- 分支 A: Crisp Path ---
        crisp_feat = self.crisp_stream(x)
        
        # --- 分支 B: Fuzzy Path ---
        # 1. 预处理
        x_fuzzy_pre = self.pre_fuzzy_conv(x)
        # 2. 模糊化 (Fuzzification) - 核心并行步骤
        x_fuzzy_membership = self.fuzzy_layer(x_fuzzy_pre)
        # 3. 模糊特征提取
        fuzzy_feat = self.fuzzy_stream_conv(x_fuzzy_membership)
        
        # --- 融合 (Fusion) ---
        # 确保两个分支的空间维度一致
        if crisp_feat.shape[2:] != fuzzy_feat.shape[2:]:
            fuzzy_feat = F.interpolate(fuzzy_feat, size=crisp_feat.shape[2:])
            
        # 拼接 (Concatenate)
        combined_feat = torch.cat([crisp_feat, fuzzy_feat], dim=1)
        
        # 全局池化
        pooled_feat = self.global_pool(combined_feat)
        
        # 分类
        logits = self.classifier(pooled_feat)
        
        return logits


def get_hpfcnn_model(num_classes=None, in_channels=3):
    """
    获取 HP-FCNN 模型实例
    
    Args:
        num_classes: 分类数量
        in_channels: 输入通道数
    
    Returns:
        HP_FCNN 模型实例
    """
    return HP_FCNN(num_classes=num_classes, in_channels=in_channels)


# =========================================================================
# 模型工厂函数 (Model Factory)
# =========================================================================

def get_model(model_type=None, num_classes=None, in_channels=3):
    """
    获取模型实例的工厂函数
    
    Args:
        model_type: 模型类型，如果不提供则使用配置中的 MODEL_TYPE
        num_classes: 分类数量
        in_channels: 输入通道数
    
    Returns:
        模型实例
    """
    if model_type is None:
        model_type = cfg.MODEL_TYPE
    
    if model_type == 'DFM_FNCN':
        return FullModel()
    elif model_type == 'TRADITIONAL_CNN':
        return TraditionalCNNModel()
    elif model_type == 'PLANTNET_ANFIS':
        return get_plantnet_model('ANFIS', num_classes=num_classes, in_channels=in_channels)
    elif model_type == 'PLANTNET_SIMPLE':
        return get_plantnet_model('SIMPLE', num_classes=num_classes, in_channels=in_channels)
    elif model_type == 'OSTEONET':
        return get_osteonet_model('resnet18', num_classes=num_classes)
    elif model_type == 'HP_FCNN':
        return get_hpfcnn_model(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")