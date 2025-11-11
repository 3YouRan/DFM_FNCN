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
DECODER_EPOCHS = 30  # [推荐] 这是一个更深的网络，需要更多训练时间
DECODER_LR = 0.001


# ==========================================
# 1. 定义解码器 (Decoder) 结构
# [关键修复] 严格遵循论文的“上采样-卷积-卷积”模式
# ==========================================
class Decoder(nn.Module):
    """
    一个更深、更强大的解码器，模仿论文 Fig. 4 的设计模式。
    将 32x7x7 的特征图上采样回 1x28x28 的图像
    """

    def __init__(self):
        super(Decoder, self).__init__()

        # --- 1. 瓶颈处理 (模拟论文中的 Residual Blocks) ---
        # 论文在 6x6x256 上运行了 6 个残差块。
        # 我们在 7x7x32 上运行几个卷积块。
        self.bottleneck = nn.Sequential(
            nn.Conv2d(cfg.N_CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # --- 2. 上采样模块 1 (7x7 -> 14x14) ---
        # 论文模式: D3 -> C3 -> C3 ...
        self.upsample1 = nn.Sequential(
            # (D3) 上采样: 64 -> 128
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # (C3) 细化
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # --- 3. 上采样模块 2 (14x14 -> 28x28) ---
        # 论文模式: D3 -> C3 ...
        self.upsample2 = nn.Sequential(
            # (D3) 上采样: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # (C3) 细化
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # --- 4. 最终输出层 ---
        # 论文模式: ... D3 (输出 2 通道)
        # 我们的模式: ... C3 (输出 1 通道, Tanh)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Tanh 匹配我们 [-1, 1] 的剪影目标
        )

    def forward(self, x):
        # x: [B, 32, 7, 7]
        x = self.bottleneck(x)  # -> [B, 64, 7, 7]
        x = self.upsample1(x)  # -> [B, 128, 14, 14]
        x = self.upsample2(x)  # -> [B, 64, 28, 28]
        x = self.final_conv(x)  # -> [B, 1, 28, 28]
        return x


# ==========================================
# 2. 定义自编码器 (Autoencoder) (保持不变)
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction


# ==========================================
# 3. 辅助函数 (保持不变)
# ==========================================
def get_train_loader():
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
# 4. 主训练循环 (保持不变)
# ==========================================
def main():
    cfg.print_config()

    print(f"正在从 {MODEL_PATH} 加载预训练的编码器...")
    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE), strict=False)

    autoencoder = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)

    for param in autoencoder.encoder.parameters():
        param.requires_grad = False
    print("编码器已冻结。开始训练解码器...")

    train_loader = get_train_loader()

    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.L1Loss()

    autoencoder.train()
    for epoch in range(DECODER_EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(cfg.DEVICE)

            # 动态创建剪影目标
            target_silhouette = (data > -1.0).float() * 2.0 - 1.0

            optimizer.zero_grad()
            reconstructed_images = autoencoder(data)
            loss = criterion(reconstructed_images, target_silhouette)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"[Epoch {epoch + 1}/{DECODER_EPOCHS}] Step {batch_idx + 1}/{len(train_loader)} | "
                      f"剪影重建损失 (L1): {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n==> Epoch {epoch + 1} 完成. 平均重建损失: {avg_loss:.6f}\n")

    torch.save(autoencoder.decoder.state_dict(), DECODER_SAVE_PATH)
    print(f"解码器训练完成！权重已保存至: {DECODER_SAVE_PATH}")


if __name__ == '__main__':
    main()