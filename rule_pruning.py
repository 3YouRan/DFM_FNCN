"""
独立规则修剪模块
从 train.py 中提取的 perform_rule_pruning 函数，用于测试和独立使用。
"""

import torch
import numpy as np
import scipy.special
import logging
from typing import Optional, Union, List, Tuple
import config as cfg  # 默认使用项目配置

def perform_rule_pruning(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    pruning_method: Optional[str] = None,
    pruning_threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int]:
    """
    执行规则修剪（独立版本）

    参数:
        model: 包含 classifier 的模型（必须是 DFM_FNCN 类型）
        test_loader: 测试数据加载器，用于计算规则激活度
        pruning_method: 修剪方法，可选 'CONSEQUENT' 或 'ACTIVATION'，默认为 cfg.PRUNING_METHOD
        pruning_threshold: 修剪阈值，默认为 cfg.PRUNING_THRESHOLD
        device: 计算设备，默认为 cfg.DEVICE
        logger: 日志记录器，可选

    返回:
        (原始规则数, 修剪后规则数)
    """
    # 使用默认配置（如果未提供）
    if pruning_method is None:
        pruning_method = cfg.PRUNING_METHOD
    if pruning_threshold is None:
        pruning_threshold = cfg.PRUNING_THRESHOLD
    if device is None:
        device = cfg.DEVICE

    log_msg = f"\n>>> 正在执行规则修剪 (Method: {pruning_method}, Th: {pruning_threshold})..."
    print(log_msg)
    if logger:
        logger.info(log_msg)

    model.eval()
    classifier = model.classifier
    num_rules = classifier.num_active_rules.item()

    if num_rules == 0:
        log_msg = "没有激活的规则，跳过修剪。"
        print(log_msg)
        if logger:
            logger.info(log_msg)
        return 0, 0

    keep_indices = []

    if pruning_method == 'CONSEQUENT':
        consequents = classifier.consequents[:num_rules].detach().cpu().numpy()
        consequents_prob = scipy.special.softmax(consequents, axis=1)
        max_probs = np.max(consequents_prob, axis=1)

        for i in range(num_rules):
            if max_probs[i] >= pruning_threshold:
                keep_indices.append(i)
            else:
                prune_msg = f"    [Prune] Rule {i} dropped (Max Prob: {max_probs[i]:.4f} < {pruning_threshold})"
                print(prune_msg)
                if logger:
                    logger.info(prune_msg)

    elif pruning_method == 'ACTIVATION':
        log_msg = "正在计算测试集上的规则激活度..."
        print(log_msg)
        if logger:
            logger.info(log_msg)

        total_activations = torch.zeros(num_rules, device=device)
        total_samples = 0

        with torch.no_grad():
            for data_tuple in test_loader:
                data = data_tuple[0].to(device)
                features = model.extractor(data)
                phi = classifier.get_rule_activations(features)
                if phi is not None:
                    total_activations += torch.sum(phi, dim=0)
                    total_samples += data.size(0)

        avg_activations = (total_activations / total_samples).cpu().numpy()

        for i in range(num_rules):
            if avg_activations[i] >= pruning_threshold:
                keep_indices.append(i)
            else:
                prune_msg = f"    [Prune] Rule {i} dropped (Avg Act: {avg_activations[i]:.5f} < {pruning_threshold})"
                print(prune_msg)
                if logger:
                    logger.info(prune_msg)

    else:
        log_msg = f"未知的修剪方法: {pruning_method}"
        print(log_msg)
        if logger:
            logger.warning(log_msg)
        return num_rules, num_rules

    if len(keep_indices) < num_rules:
        classifier.prune_rules(keep_indices)
        log_msg = f"规则修剪完成，从 {num_rules} 条规则修剪到 {len(keep_indices)} 条。\n"
    else:
        log_msg = "没有规则被修剪。\n"

    print(log_msg)
    if logger:
        logger.info(log_msg)

    return num_rules, len(keep_indices)


def perform_rule_pruning_with_config(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: Optional[object] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int]:
    """
    使用配置对象执行规则修剪（更灵活的版本）

    参数:
        model: 包含 classifier 的模型
        test_loader: 测试数据加载器
        config: 包含 PRUNING_METHOD, PRUNING_THRESHOLD, DEVICE 等属性的配置对象
                如果为 None，则使用默认的 cfg
        logger: 日志记录器

    返回:
        (原始规则数, 修剪后规则数)
    """
    if config is None:
        config = cfg

    pruning_method = getattr(config, 'PRUNING_METHOD', 'CONSEQUENT')
    pruning_threshold = getattr(config, 'PRUNING_THRESHOLD', 0.5)
    device = getattr(config, 'DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return perform_rule_pruning(
        model=model,
        test_loader=test_loader,
        pruning_method=pruning_method,
        pruning_threshold=pruning_threshold,
        device=device,
        logger=logger
    )


if __name__ == '__main__':
    # 简单的自测代码
    print("规则修剪模块已加载。")
    print("请导入 perform_rule_pruning 函数使用。")
    print("示例:")
    print("    from rule_pruning import perform_rule_pruning")
    print("    original, pruned = perform_rule_pruning(model, test_loader)")