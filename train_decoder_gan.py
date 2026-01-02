import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import sys
import config as cfg
from models import FullModel
from train_decoder import get_decoder, get_train_loader  # 复用之前的工具函数

# --- 配置 ---
RUN_DIR_TO_LOAD = './checkpoints/！！！GTSRB_DFM_FNCN_RESNET18_PRETRAINED_20251209_115250'
# ---------------------------------------------------------

MODEL_PATH = os.path.join(RUN_DIR_TO_LOAD, 'best_model.pth')
# 保存为 perceptual 版本，以免覆盖之前的
DECODER_SAVE_PATH = os.path.join(RUN_DIR_TO_LOAD, 'decoder_perceptual.pth')
EPOCHS = 200
LR = 0.001


# ======================================================
# 1. 定义感知损失模块 (Perceptual Loss)
# ======================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # 加载预训练的 VGG16
        try:
            weights = models.VGG16_Weights.DEFAULT
            vgg = models.vgg16(weights=weights).features
        except AttributeError:
            vgg = models.vgg16(pretrained=True).features

        # 我们只使用 VGG 的前几层来提取纹理和形状特征
        # 截取到第 16 层 (ReLU3_3) 包含了丰富的边缘和纹理信息
        self.blocks = nn.Sequential(*list(vgg.children())[:16]).eval()

        # 冻结参数，不参与训练
        for param in self.blocks.parameters():
            param.requires_grad = False

        self.blocks = self.blocks.to(cfg.DEVICE)

        # VGG 标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(cfg.DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(cfg.DEVICE)

    def forward(self, input, target):
        # 1. 输入数据处理
        # 我们的数据在 [-1, 1] (Tanh输出)，需要转换到 [0, 1]
        input = (input + 1) / 2
        target = (target + 1) / 2

        # 2. 通道适配
        # VGG 需要 3 通道输入。如果是灰度图 (1通道)，重复3次
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 3. 标准化 (VGG 预处理要求)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # 4. 提取特征
        x_feat = self.blocks(input)
        y_feat = self.blocks(target)

        # 5. 计算特征空间的 MSE 损失
        return F.mse_loss(x_feat, y_feat)


# ======================================================
# 2. 自动配置工具
# ======================================================
def configure_model_from_checkpoint(checkpoint):
    """从检查点恢复配置"""
    if 'config_params' not in checkpoint:
        print("错误: 无法从检查点恢复配置。")
        sys.exit()
    params = checkpoint['config_params']
    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.TARGET_SIZE = config_data['target_size']

    # 假设所有提取器都输出 6x6
    cfg.IMG_DIM_OUT = 6
    cfg.N_CHANNELS_OUT = 128

    print(f"[Perceptual Training] Config Loaded: {cfg.DATASET_NAME}, {cfg.EXTRACTOR_TYPE}")


# ======================================================
# 3. 主训练循环
# ======================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        sys.exit()

    # 1. 加载配置和编码器
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.DEVICE)
    configure_model_from_checkpoint(checkpoint)

    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = base_model.extractor
    encoder.requires_grad_(False)  # 冻结编码器
    encoder.eval()

    # 2. 初始化解码器 (使用之前的对称结构)
    decoder = get_decoder().to(cfg.DEVICE)

    # 3. 损失函数
    # 组合损失：像素损失 (L1) + 感知损失 (Perceptual)
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss()

    # 4. 优化器
    optimizer = optim.Adam(decoder.parameters(), lr=LR)

    # 5. 数据加载
    train_loader = get_train_loader()

    print(f"开始感知解码器训练 (Perceptual Decoder)... Epochs: {EPOCHS}")

    decoder.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pixel_loss_sum = 0
        perc_loss_sum = 0

        for i, data_tuple in enumerate(train_loader):
            # 适配不同数据集格式
            if isinstance(data_tuple, list):
                imgs = data_tuple[0]
            else:
                imgs = data_tuple[0]

            imgs = imgs.to(cfg.DEVICE)

            # 获取隐变量
            with torch.no_grad():
                features = encoder(imgs)

            optimizer.zero_grad()

            # 重建图像
            recon_imgs = decoder(features)

            # 计算损失
            # 1. 像素级差异 (保证颜色/亮度大体正确)
            l_pixel = criterion_pixel(recon_imgs, imgs)

            # 2. 感知级差异 (保证结构/纹理清晰)
            # 权重 0.1~0.5 通常效果较好，取决于 VGG 特征的数值范围
            l_perc = criterion_perceptual(recon_imgs, imgs)

            # 总损失
            loss = l_pixel + 0.2 * l_perc

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pixel_loss_sum += l_pixel.item()
            perc_loss_sum += l_perc.item()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{EPOCHS}] [Batch {i}] Loss: {loss.item():.4f} (Pixel: {l_pixel.item():.4f}, Perc: {l_perc.item():.4f})")

        print(f"==> Epoch {epoch} 完成. Avg Loss: {total_loss / len(train_loader):.4f}")

    # 6. 保存
    torch.save(decoder.state_dict(), DECODER_SAVE_PATH)
    print(f"感知解码器训练完成！权重已保存至: {DECODER_SAVE_PATH}")
    print("现在可以修改 visualize_rule.py，将 DECODER_PATH 指向 'decoder_perceptual.pth' 来查看效果。")


if __name__ == '__main__':
    main()