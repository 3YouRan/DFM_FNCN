"""
测试 OsteoNet 模型是否能正确运行
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

import config as cfg
from models import OsteoNet, OsteoNetSimple, get_osteonet_model

def test_osteonet_models():
    """测试 OsteoNet 各种配置"""
    print("=" * 60)
    print("测试 OsteoNet 模型")
    print("=" * 60)
    
    # 设置设备
    device = cfg.DEVICE
    print(f"设备: {device}")
    print(f"输入通道数: {cfg.IN_CHANNELS}")
    print(f"类别数: {cfg.N_CLASSES}")
    print(f"输入尺寸: {cfg.TARGET_SIZE}")
    print("-" * 60)
    
    # 创建测试输入 (batch=2, channels, height, width)
    if cfg.IN_CHANNELS == 1:
        test_input = torch.rand(2, 1, cfg.TARGET_SIZE[0], cfg.TARGET_SIZE[1])
    else:
        test_input = torch.rand(2, cfg.IN_CHANNELS, cfg.TARGET_SIZE[0], cfg.TARGET_SIZE[1])
    
    print(f"测试输入形状: {test_input.shape}")
    
    # 测试 OsteoNetSimple (不需要预训练权重)
    print("\n--- 测试 OsteoNetSimple ---")
    try:
        model_simple = OsteoNetSimple(num_classes=cfg.N_CLASSES, in_channels=cfg.IN_CHANNELS)
        model_simple.eval()
        with torch.no_grad():
            logits, enhanced = model_simple(test_input)
        print(f"  logits 形状: {logits.shape}")
        print(f"  enhanced 形状: {enhanced.shape}")
        print(f"  预测概率形状: {torch.softmax(logits, dim=1).shape}")
        print("  [✓] OsteoNetSimple 测试通过!")
    except Exception as e:
        print(f"  [✗] OsteoNetSimple 测试失败: {e}")
    
    # 测试 OsteoNet (使用 ResNet50)
    print("\n--- 测试 OsteoNet (ResNet50) ---")
    try:
        model_resnet50 = OsteoNet(model_name='resnet50', num_classes=cfg.N_CLASSES, pretrained=False)
        model_resnet50.eval()
        with torch.no_grad():
            logits, enhanced = model_resnet50(test_input)
        print(f"  logits 形状: {logits.shape}")
        print(f"  enhanced 形状: {enhanced.shape}")
        print(f"  预测概率形状: {torch.softmax(logits, dim=1).shape}")
        print("  [✓] OsteoNet (ResNet50) 测试通过!")
    except Exception as e:
        print(f"  [✗] OsteoNet (ResNet50) 测试失败: {e}")
    
    # 测试 OsteoNet (使用 ResNet18)
    print("\n--- 测试 OsteoNet (ResNet18) ---")
    try:
        model_resnet = OsteoNet(model_name='resnet18', num_classes=cfg.N_CLASSES, pretrained=False)
        model_resnet.eval()
        with torch.no_grad():
            logits, enhanced = model_resnet(test_input)
        print(f"  logits 形状: {logits.shape}")
        print(f"  enhanced 形状: {enhanced.shape}")
        print(f"  预测概率形状: {torch.softmax(logits, dim=1).shape}")
        print("  [✓] OsteoNet (ResNet18) 测试通过!")
    except Exception as e:
        print(f"  [✗] OsteoNet (ResNet18) 测试失败: {e}")
    
    # 测试 OsteoNet (使用 MobileNetV2)
    print("\n--- 测试 OsteoNet (MobileNetV2) ---")
    try:
        model_mobilenet = OsteoNet(model_name='mobilenet_v2', num_classes=cfg.N_CLASSES, pretrained=False)
        model_mobilenet.eval()
        with torch.no_grad():
            logits, enhanced = model_mobilenet(test_input)
        print(f"  logits 形状: {logits.shape}")
        print(f"  enhanced 形状: {enhanced.shape}")
        print(f"  预测概率形状: {torch.softmax(logits, dim=1).shape}")
        print("  [✓] OsteoNet (MobileNetV2) 测试通过!")
    except Exception as e:
        print(f"  [✗] OsteoNet (MobileNetV2) 测试失败: {e}")
    
    # 测试 get_osteonet_model 工厂函数
    print("\n--- 测试 get_osteonet_model 工厂函数 ---")
    try:
        model_factory = get_osteonet_model(model_type='simple', num_classes=cfg.N_CLASSES)
        print(f"  [✓] get_osteonet_model (simple) 创建成功!")
    except Exception as e:
        print(f"  [✗] get_osteonet_model (simple) 失败: {e}")
    
    try:
        model_factory = get_osteonet_model(model_type='resnet50', num_classes=cfg.N_CLASSES)
        print(f"  [✓] get_osteonet_model (resnet50) 创建成功!")
    except Exception as e:
        print(f"  [✗] get_osteonet_model (resnet50) 失败: {e}")
    
    try:
        model_factory = get_osteonet_model(model_type='resnet18', num_classes=cfg.N_CLASSES)
        print(f"  [✓] get_osteonet_model (resnet18) 创建成功!")
    except Exception as e:
        print(f"  [✗] get_osteonet_model (resnet18) 失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_fuzzy_enhancement():
    """测试模糊增强层"""
    print("\n" + "=" * 60)
    print("测试 FuzzyContrastEnhancement 层")
    print("=" * 60)
    
    from models import FuzzyContrastEnhancement
    
    # 测试单通道输入
    print("\n--- 单通道输入测试 ---")
    gray_input = torch.rand(2, 1, 28, 28)  # 值域 [0, 1]
    fuzzy_layer = FuzzyContrastEnhancement()
    enhanced = fuzzy_layer(gray_input)
    print(f"  输入形状: {gray_input.shape}, 值域: [{gray_input.min():.4f}, {gray_input.max():.4f}]")
    print(f"  输出形状: {enhanced.shape}, 值域: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
    print(f"  输入 Std: {gray_input.std():.4f}, 输出 Std: {enhanced.std():.4f}")
    print("  [✓] 单通道测试通过!" if enhanced.shape == gray_input.shape else "  [✗] 形状不匹配!")
    
    # 测试三通道输入
    print("\n--- 三通道输入测试 ---")
    rgb_input = torch.rand(2, 3, 28, 28)  # 值域 [0, 1]
    enhanced = fuzzy_layer(rgb_input)
    print(f"  输入形状: {rgb_input.shape}, 值域: [{rgb_input.min():.4f}, {rgb_input.max():.4f}]")
    print(f"  输出形状: {enhanced.shape}, 值域: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
    print(f"  输入 Std: {rgb_input.std():.4f}, 输出 Std: {enhanced.std():.4f}")
    print("  [✓] 三通道测试通过!" if enhanced.shape == rgb_input.shape else "  [✗] 形状不匹配!")
    
    # 测试 0-255 值域输入
    print("\n--- 0-255 值域输入测试 ---")
    uint8_like_input = torch.rand(2, 1, 28, 28) * 255
    enhanced = fuzzy_layer(uint8_like_input)
    print(f"  输入形状: {uint8_like_input.shape}, 值域: [{uint8_like_input.min():.2f}, {uint8_like_input.max():.2f}]")
    print(f"  输出形状: {enhanced.shape}, 值域: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
    print("  [✓] 0-255 值域测试通过!" if enhanced.max() <= 1.0 else "  [✗] 输出值域错误!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_fuzzy_enhancement()
    test_osteonet_models()
