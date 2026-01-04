import torch
import sys
sys.path.insert(0, '.')
from models import Dynamic_DFM_FNCN
import config as cfg

# 覆盖配置以便测试
cfg.MAX_RULES = 10
cfg.N_CHANNELS_OUT = 4
cfg.P_DIM = 9
cfg.N_CLASSES = 5
cfg.USE_ATTENTION = False
cfg.USE_RULE_MERGING = True
cfg.MERGING_METHOD = 'SIMILARITY'
cfg.MERGING_THRESHOLD = 0.9
cfg.MERGING_STRATEGY = 'WEIGHTED_AVERAGE'

print("创建模型...")
model = Dynamic_DFM_FNCN(n_channels=cfg.N_CHANNELS_OUT, p_dim=cfg.P_DIM, n_classes=cfg.N_CLASSES, max_rules=cfg.MAX_RULES)
print(f"初始 max_rules: {model.max_rules}")
print(f"centers 形状: {model.centers.shape}")
print(f"widths_param 形状: {model.widths_param.shape}")
print(f"consequents 形状: {model.consequents.shape}")

# 初始化一些规则
with torch.no_grad():
    model.centers[:3] = torch.randn(3, cfg.N_CHANNELS_OUT, cfg.P_DIM)
    model.consequents[:3, :] = torch.randn(3, cfg.N_CLASSES)
    model.num_active_rules.fill_(3)

print(f"激活规则数: {model.num_active_rules.item()}")

# 模拟融合：创建两个非常相似的规则，使它们被融合
# 我们手动设置相似度高的中心
center1 = torch.randn(cfg.N_CHANNELS_OUT, cfg.P_DIM)
center2 = center1 + 0.01 * torch.randn_like(center1)  # 非常相似
with torch.no_grad():
    model.centers[0] = center1
    model.centers[1] = center2
    model.consequents[0] = torch.tensor([1., 0., 0., 0., 0.])
    model.consequents[1] = torch.tensor([1., 0., 0., 0., 0.])  # 相同类别
    model.num_active_rules.fill_(2)

print("执行规则融合...")
merged_count, merge_pairs = model.merge_similar_rules(test_loader=None)
print(f"融合了 {merged_count} 对规则")
print(f"融合后激活规则数: {model.num_active_rules.item()}")
print(f"融合后 max_rules: {model.max_rules}")
print(f"融合后 centers 形状: {model.centers.shape}")
print(f"融合后 widths_param 形状: {model.widths_param.shape}")
print(f"融合后 consequents 形状: {model.consequents.shape}")

# 检查形状是否减小
if model.centers.shape[0] < cfg.MAX_RULES:
    print("成功：参数张量大小已减小，显存占用将减少。")
else:
    print("失败：参数张量大小未减小。")

# 检查显存占用（可选）
if torch.cuda.is_available():
    model.cuda()
    print(f"CUDA 内存分配: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # 注意：由于 PyTorch 缓存，实际释放的内存可能不会立即反映。