"""
HP-FCNN 模型测试脚本
用于验证模型集成是否正确
"""
import torch
import sys
sys.path.append('.')

import config as cfg
from models import HP_FCNN, get_model

def test_hpfcnn_shapes():
    """测试 HP-FCNN 模型形状"""
    print("=" * 50)
    print("测试 HP-FCNN 模型形状")
    print("=" * 50)
    
    # 配置
    BATCH_SIZE = 2
    CHANNELS = cfg.IN_CHANNELS
    HEIGHT, WIDTH = cfg.TARGET_SIZE
    CLASSES = cfg.N_CLASSES
    
    print(f"输入配置: 通道={CHANNELS}, 尺寸=({HEIGHT}x{WIDTH}), 类别={CLASSES}")
    
    # 实例化模型
    model = HP_FCNN(num_classes=CLASSES, in_channels=CHANNELS)
    print(f"\n>>> 模型实例化成功: {model.__class__.__name__}")
    
    # 模拟输入数据
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"\n>>> 形状检查:")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 简单断言
    expected_shape = (BATCH_SIZE, CLASSES)
    assert output.shape == expected_shape, f"输出维度错误！期望 {expected_shape}, 实际 {output.shape}"
    print("✅ 测试通过：HP-FCNN 架构运行正常。")
    
    return True

def test_model_factory():
    """测试模型工厂函数"""
    print("\n" + "=" * 50)
    print("测试模型工厂函数")
    print("=" * 50)
    
    # 保存原始配置
    original_type = cfg.MODEL_TYPE
    
    try:
        # 测试通过配置获取模型
        cfg.MODEL_TYPE = 'HP_FCNN'
        model = get_model()
        print(f"通过配置获取模型: {model.__class__.__name__}")
        
        # 测试直接指定类型获取模型
        model2 = get_model(model_type='HP_FCNN')
        print(f"直接指定类型获取模型: {model2.__class__.__name__}")
        
        assert isinstance(model, HP_FCNN), "模型类型错误"
        print("✅ 测试通过：模型工厂函数工作正常。")
        
        return True
    finally:
        # 恢复原始配置
        cfg.MODEL_TYPE = original_type

def test_hpfcnn_components():
    """测试 HP-FCNN 组件"""
    print("\n" + "=" * 50)
    print("测试 HP-FCNN 组件")
    print("=" * 50)
    
    from models import FuzzyLayer
    
    # 测试 FuzzyLayer
    in_channels = 32
    num_fuzzy_sets = 4
    fuzzy_layer = FuzzyLayer(in_channels, num_fuzzy_sets)
    
    # 模拟输入
    dummy_input = torch.randn(2, in_channels, 28, 28)
    
    # 前向传播
    output = fuzzy_layer(dummy_input)
    
    expected_channels = in_channels * num_fuzzy_sets  # 32 * 4 = 128
    assert output.shape == (2, expected_channels, 28, 28), f"FuzzyLayer 输出形状错误: {output.shape}"
    print(f"FuzzyLayer 测试通过: 输入 {dummy_input.shape} -> 输出 {output.shape}")
    
    return True

if __name__ == "__main__":
    print("开始 HP-FCNN 模型测试...\n")
    
    all_passed = True
    
    try:
        all_passed &= test_hpfcnn_components()
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_hpfcnn_shapes()
    except Exception as e:
        print(f"❌ 形状测试失败: {e}")
        all_passed = False
    
    try:
        all_passed &= test_model_factory()
    except Exception as e:
        print(f"❌ 工厂函数测试失败: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！HP-FCNN 模型集成成功。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    print("=" * 50)
