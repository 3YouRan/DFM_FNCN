import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg


class SimpleCNN(nn.Module):
    """
    一个简单的 DCNN 特征提取器。
    输入: [B, 1, 28, 28] (Fashion-MNIST)
    输出: [B, N_CHANNELS, 7, 7]
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: 28x28 -> 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Layer 2: 14x14 -> 7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, cfg.N_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Dynamic_DFM_FNCN(nn.Module):
    """
    论文核心：支持在线动态规则生成的 DFM-FNCN 层。
    """

    def __init__(self, n_channels, p_dim, n_classes, max_rules=cfg.MAX_RULES, phi_th=cfg.PHI_TH):
        super(Dynamic_DFM_FNCN, self).__init__()
        self.n_channels = n_channels
        self.p_dim = p_dim
        self.n_classes = n_classes
        self.max_rules = max_rules
        self.phi_th = phi_th

        # 当前活跃的规则数量 (初始为0，由数据驱动增长)
        self.num_active_rules = 0

        # 批归一化层，用于稳定输入特征图的分布
        self.bn = nn.BatchNorm2d(n_channels)

        # --- 预分配最大容量的参数空间 ---
        # 我们使用 nn.Parameter，但实际上只有前 num_active_rules 个会被使用和更新。
        # Centers: 规则的前件中心特征图 C_j^i
        self.centers = nn.Parameter(torch.zeros(max_rules, n_channels, p_dim))
        # Widths: 规则的前件宽度 sigma_j^i (存储其原始值，使用 softplus 激活)
        self.widths_param = nn.Parameter(torch.ones(max_rules, n_channels) * cfg.INIT_SIGMA)
        # Consequents: 规则的后件权重 (Zero-order TSK)
        self.consequents = nn.Parameter(torch.zeros(max_rules, n_classes))

    def forward(self, x, labels=None, training_phase=False):
        """
        前向传播。
        如果 training_phase=True 且提供了 labels，则会执行在线规则生成检查。
        """
        b = x.size(0)
        # 1. 特征图归一化
        x = self.bn(x)
        # 2. 展平空间维度: [B, N, 7, 7] -> [B, N, 49]
        x_flat = x.view(b, self.n_channels, -1)

        # --- 处理初始状态 (无规则时) ---
        if self.num_active_rules == 0:
            if training_phase and labels is not None:
                # 如果是训练初始阶段，强制添加第一条规则
                self._add_rule(x_flat[0], labels[0])
            else:
                # 如果是测试阶段且无规则，返回全零 (虽然这种情况在合理流程下不应发生)
                return torch.zeros(b, self.n_classes).to(x.device)

        # --- 提取当前活跃的规则参数 ---
        # 使用切片操作确保只计算活跃规则，梯度也只回传给它们
        active_centers = self.centers[:self.num_active_rules]  # [R_active, N, P]
        active_widths_param = self.widths_param[:self.num_active_rules]  # [R_active, N]
        active_consequents = self.consequents[:self.num_active_rules]  # [R_active, n_classes]

        # --- 模糊化 (Fuzzification) ---
        # x_exp: [B, 1, N, P]
        # c_exp: [1, R_active, N, P]
        x_exp = x_flat.unsqueeze(1)
        c_exp = active_centers.unsqueeze(0)

        # 计算匹配度 (Matching Degree): 余弦相似度, 沿 P 维度(dim=3)
        M = F.cosine_similarity(x_exp, c_exp, dim=3)  # -> [B, R_active, N]

        # 计算距离 d = 1 - M
        d = 1.0 - M

        # 计算宽度 sigma (确保为正数)
        sigma = F.softplus(active_widths_param).unsqueeze(0) + 1e-6  # -> [1, R_active, N]

        # 计算高斯隶属度 mu
        mu = torch.exp(-torch.pow(d, 2) / torch.pow(sigma, 2))  # -> [B, R_active, N]

        # --- 规则激发 (Firing Strength) ---
        # 使用乘积算子聚合所有通道的隶属度
        phi = torch.prod(mu, dim=2)  # -> [B, R_active]

        # --- 在线规则生成 (Online Rule Generation) ---
        # 仅在训练阶段进行
        if training_phase and labels is not None and self.num_active_rules < self.max_rules:
            # 计算每个样本在当前所有规则下的最大激发强度
            max_phi_per_sample, _ = phi.max(dim=1)  # [B]

            # 找出覆盖不足的样本 (激发强度 < 阈值)
            poorly_covered_mask = max_phi_per_sample < self.phi_th

            if poorly_covered_mask.any():
                # 选取第一个覆盖不足的样本作为新规则的种子
                # (为了训练稳定性，每个 batch 最多只添加一条规则)
                idx_to_add = torch.nonzero(poorly_covered_mask)[0].item()
                self._add_rule(x_flat[idx_to_add], labels[idx_to_add])
                # 注意：新规则在当前 batch 不会立即生效，而是在下一个 batch 生效。

        # --- 去模糊化 (Defuzzification) ---
        # 归一化激发强度
        phi_sum = torch.sum(phi, dim=1, keepdim=True) + 1e-8
        phi_norm = phi / phi_sum  # [B, R_active]

        # 计算加权平均输出 (Logits)
        output_logits = torch.matmul(phi_norm, active_consequents)  # [B, n_classes]

        return output_logits

    def _add_rule(self, center_feat, label):
        """
        内部方法：向规则库添加一条新规则。
        """
        # 在 no_grad 环境下初始化新参数，避免影响当前的计算图
        with torch.no_grad():
            new_idx = self.num_active_rules

            # 1. 初始化中心: 使用当前样本的特征图
            self.centers[new_idx].copy_(center_feat.detach())

            # 2. 初始化宽度: 使用预设的初始值
            # 计算 softplus 的逆，以便前向传播时 softplus(param) 等于我们想要的 INIT_SIGMA
            init_val = np.log(np.exp(cfg.INIT_SIGMA) - 1)
            self.widths_param[new_idx].fill_(init_val)

            # 3. 初始化后件: 使用 One-hot 编码，强烈指向当前样本的真实类别
            one_hot = torch.zeros(self.n_classes, device=self.centers.device)
            one_hot[label] = 1.0  # 将目标类别的初始权重设为 1.0
            self.consequents[new_idx].copy_(one_hot)

            self.num_active_rules += 1
            # 打印日志 (可选，用于调试规则增长过程)
            # print(f"[Dynamic Rule] Added rule #{self.num_active_rules} for class {label.item()}")


class FullModel(nn.Module):
    """
    将特征提取器和分类器组合在一起的完整模型。
    """

    def __init__(self):
        super(FullModel, self).__init__()
        self.extractor = SimpleCNN()
        self.classifier = Dynamic_DFM_FNCN(
            n_channels=cfg.N_CHANNELS,
            p_dim=cfg.P_DIM,
            n_classes=cfg.N_CLASSES
        )

    def forward(self, x, labels=None, training_phase=False):
        # 1. 提取特征
        features = self.extractor(x)
        # 2. 分类 (并可能触发规则生成)
        logits = self.classifier(features, labels=labels, training_phase=training_phase)
        return logits