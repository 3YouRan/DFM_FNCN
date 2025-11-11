import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 导入我们的自定义模块
import config as cfg
from models import FullModel  # 我们需要 FullModel 来加载编码器

# --- 配置 ---
DECODER_SAVE_PATH = os.path.join('./checkpoints', 'decoder.pth')
MODEL_PATH = os.path.join('./checkpoints', 'best_model.pth')
DECODER_EPOCHS = 10  # 训练解码器通常很快，10-20 个 epoch 足够
DECODER_LR = 0.001


# ==========================================
# 1. 定义解码器 (Decoder) 结构
# ==========================================
class Decoder(nn.Module):
    """
    将 32x7x7 的特征图上采样回 1x28x28 的图像
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.upsample_layers = nn.Sequential(
            # 输入: [B, 32, 7, 7]
            nn.ConvTranspose2d(cfg.N_CHANNELS, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # -> [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # -> [B, 128, 28, 28]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # 投影回 1 个通道
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            # -> [B, 1, 28, 28]

            # 使用 Tanh 激活函数，因为原始图像被归一化到 [-1, 1]
            nn.Tanh()
        )

    def forward(self, x):
        return self.upsample_layers(x)


# ==========================================
# 2. 定义自编码器 (Autoencoder)
# ==========================================
class Autoencoder(nn.Module):
    """
    组合 冻结的编码器 和 可训练的解码器
    """

    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, x):
        # 编码器将 [1, 28, 28] -> [32, 7, 7]
        features = self.encoder(x)
        # 解码器将 [32, 7, 7] -> [1, 28, 28]
        reconstruction = self.decoder(features)
        return reconstruction


# ==========================================
# 3. 辅助函数
# ==========================================
def get_train_loader():
    """获取训练数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root=cfg.DATA_ROOT,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    return train_loader


# ==========================================
# 4. 主训练循环
# ==========================================
def main():
    cfg.print_config()

    # 1. 准备模型 (加载预训练编码器)
    print(f"正在从 {MODEL_PATH} 加载预训练的编码器...")
    base_model = FullModel().to(cfg.DEVICE)

    # strict=False 允许我们只加载 state_dict 中匹配的部分 (即编码器)
    # 这会忽略分类器 (classifier) 部分的缺失
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE), strict=False)

    # 封装自编码器
    autoencoder = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)

    # [重要] 冻结编码器的所有参数
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    print("编码器已冻结。开始训练解码器...")

    # 2. 准备数据
    train_loader = get_train_loader()

    # 3. 准备优化器和损失函数
    # [重要] 优化器只应包含解码器的参数
    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()  # 均方误差 (MSE) 是重建的标准损失

    # 4. 训练循环
    autoencoder.train()
    for epoch in range(DECODER_EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(cfg.DEVICE)

            optimizer.zero_grad()

            # 前向传播
            reconstructed_images = autoencoder(data)

            # 计算损失 (重建图像 vs 原始图像)
            loss = criterion(reconstructed_images, data)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"[Epoch {epoch + 1}/{DECODER_EPOCHS}] Step {batch_idx + 1}/{len(train_loader)} | "
                      f"重建损失 (MSE): {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n==> Epoch {epoch + 1} 完成. 平均重建损失: {avg_loss:.6f}\n")

    # 5. 保存训练好的解码器
    torch.save(autoencoder.decoder.state_dict(), DECODER_SAVE_PATH)
    print(f"解码器训练完成！权重已保存至: {DECODER_SAVE_PATH}")


if __name__ == '__main__':
    main()