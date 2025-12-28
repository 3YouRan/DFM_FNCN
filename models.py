import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import config as cfg
# [修改] 使用 torch.amp.autocast
from torch.amp import autocast


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
            # 初始化为 0 -> Softmax 后为均匀分布 (1/N)
            self.alpha = nn.Parameter(torch.zeros(max_rules, n_channels))
        else:
            self.register_parameter('alpha', None)

    def forward(self, x, labels=None, training_phase=False):
        # [修改] 强制使用 float32/float64 避免下溢，并使用 torch.amp.autocast
        with autocast(device_type=cfg.DEVICE.type, enabled=False):
            x = x.to(torch.float32)
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
                    self.pending_rule_data = (x_flat[min_idx].detach(), labels[min_idx].detach())

            output_logits = torch.matmul(phi_norm_double.to(torch.float32), active_consequents)

        return output_logits

    def get_rule_activations(self, x):
        """[创新点 2] 获取规则的激活强度 (Phi)，用于基于激活的修剪"""
        with torch.no_grad():
            x = x.to(torch.float32)
            b = x.size(0)
            x = self.bn(x)
            x_flat = x.view(b, self.n_channels, -1)

            active_rules_count = self.num_active_rules.item()
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
            old_count = self.num_active_rules.item()
            self.num_active_rules.fill_(new_count)
            print(f"--> [Pruning] 规则数从 {old_count} 减少到 {new_count}。")

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
            self.num_active_rules.fill_(num_init)
            print(f"--> [Batch Init] 已批量初始化 {num_init} 条规则。")

    def merge_similar_rules(self, test_loader=None):
        """[创新点 6] 动态融合相似规则"""
        if not cfg.USE_RULE_MERGING:
            return 0, []

        active_rules_count = self.num_active_rules.item()
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
            old_count = self.num_active_rules.item()
            self.num_active_rules.fill_(new_count)

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
                    merged_alpha = weight_i * alpha_i + weight_j * alpha_j
                elif cfg.MERGING_STRATEGY == 'DOMINANT_RULE':
                    if conf_i >= conf_j:
                        merged_alpha = alpha_i
                    else:
                        merged_alpha = alpha_j

            merged_rules_info.append({
                'center': merged_center,
                'width': merged_width,
                'consequent': merged_consequent,
                'alpha': merged_alpha,
                'original_indices': (i, j),
                'confidences': (conf_i.item(), conf_j.item())
            })

        return merged_rules_info

    def _update_parameters_after_merging(self, merged_rules_info):
        """融合后更新模型参数"""
        if not merged_rules_info:
            return

        active_rules_count = self.num_active_rules.item()
        merge_pairs = [info['original_indices'] for info in merged_rules_info]

        # 找出需要保留的规则索引（未被融合的规则）
        all_indices = set(range(active_rules_count))
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