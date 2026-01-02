"""
测试规则修剪功能的独立脚本。
"""

import torch
import sys
import os
sys.path.append('.')  # 确保可以导入当前目录的模块

import config as cfg
from models import FullModel
from train import get_data_loaders  # 导入数据加载函数
from rule_pruning import perform_rule_pruning

def main():
    print("=== 测试规则修剪功能 ===")
    
    # 1. 检查配置
    print(f"数据集: {cfg.DATASET_NAME}")
    print(f"模型类型: {cfg.MODEL_TYPE}")
    print(f"修剪方法: {cfg.PRUNING_METHOD}")
    print(f"修剪阈值: {cfg.PRUNING_THRESHOLD}")
    print(f"设备: {cfg.DEVICE}")
    
    # 2. 加载测试数据（仅测试集）
    print("\n加载测试数据...")
    _, test_loader, _ = get_data_loaders()
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 3. 加载预训练模型
    checkpoint_path = "checkpoints/MIO_TCD_CLASSIFICATION_DFM_FNCN_RESNET18_PRETRAINED_20251231_101215/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        # 尝试查找其他检查点
        import glob
        candidates = glob.glob("checkpoints/*/best_model.pth")
        if candidates:
            checkpoint_path = candidates[0]
            print(f"使用找到的检查点: {checkpoint_path}")
        else:
            print("未找到任何检查点，退出。")
            return
    
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    
    # 创建模型实例
    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. 检查规则数量
    classifier = model.classifier
    num_rules_before = classifier.num_active_rules.item()
    print(f"修剪前规则数量: {num_rules_before}")
    
    # 5. 执行规则修剪
    print("\n开始规则修剪...")
    original, pruned = perform_rule_pruning(model, test_loader, logger=None)
    
    # 6. 验证结果
    num_rules_after = classifier.num_active_rules.item()
    print(f"\n修剪后规则数量: {num_rules_after}")
    print(f"修剪掉的规则数量: {original - pruned}")
    
    # 7. 可选：评估修剪前后的性能
    from train import evaluate
    import torch.nn as nn
    
    criterion = nn.CrossEntropyLoss()
    test_loss_before, test_acc_before = evaluate(model, test_loader, criterion)
    print(f"修剪前测试准确率: {test_acc_before:.2f}%")
    
    # 如果需要，可以保存修剪后的模型
    save_pruned = input("\n是否保存修剪后的模型？(y/n): ").strip().lower()
    if save_pruned == 'y':
        pruned_path = checkpoint_path.replace('.pth', '_pruned.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_rules': cfg.MAX_RULES,
            'config_params': checkpoint.get('config_params', {})
        }, pruned_path)
        print(f"修剪后的模型已保存至: {pruned_path}")
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()