import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import config as cfg
from models import FullModel  # 我们需要 FullModel 来加载编码器

# --- [重要] 在此处配置您要为其训练解码器的运行目录 ---
RUN_DIR_TO_LOAD = './checkpoints/DFM_FNCN_VGG16_PRETRAINED_20251111_182945'
# ---

MODEL_PATH = os.path.join(RUN_DIR_TO_LOAD, 'best_model.pth')
DECODER_SAVE_PATH = os.path.join(RUN_DIR_TO_LOAD, 'decoder.pth')
DECODER_EPOCHS = 100  # [推荐] 这是一个更深的网络，需要更多训练
DECODER_LR = 0.001


def configure_model_from_path(run_dir):
    """从目录名称中推断配置"""
    if not os.path.exists(run_dir):
        print(f"错误: 目录不存在 '{run_dir}'")
        sys.exit()
    print(f"正在从目录名中推断配置: {run_dir}")
    base_name = os.path.basename(run_dir)
    if 'DFM_FNCN' not in base_name:
        print(f"错误: {run_dir} 不是 DFM_FNCN 模型的运行目录。")
        sys.exit()
    cfg.MODEL_TYPE = 'DFM_FNCN'
    if 'RESNET18_PRETRAINED' in base_name:
        cfg.EXTRACTOR_TYPE = 'RESNET18_PRETRAINED'
    elif 'VGG16_PRETRAINED' in base_name:
        cfg.EXTRACTOR_TYPE = 'VGG16_PRETRAINED'
    elif 'SIMPLE_CNN' in base_name:
        cfg.EXTRACTOR_TYPE = 'SIMPLE_CNN'
    else:
        raise ValueError(f"无法从目录名中推断 EXTRACTOR_TYPE。'{base_name}'")
    print(f"推断配置: MODEL_TYPE={cfg.MODEL_TYPE}, EXTRACTOR_TYPE={cfg.EXTRACTOR_TYPE}")


# ======================================================
# 1. 定义对称的解码器 (Decoders)
# ======================================================

class BasicBlock(nn.Module):
    """[新] 论文中 Fig. 4 使用的残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out


class SimpleCNNDecoder(nn.Module):
    """对称于 SimpleCNN, 在瓶颈处添加了处理块"""

    def __init__(self):
        super(SimpleCNNDecoder, self).__init__()

        # [新] 瓶颈处理块
        self.bottleneck = nn.Sequential(
            BasicBlock(cfg.N_CHANNELS, cfg.N_CHANNELS),
            BasicBlock(cfg.N_CHANNELS, cfg.N_CHANNELS)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bottleneck(x)  # [新] 先处理
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


class ResNet18Decoder(nn.Module):
    """对称于 ResNet18, 严格遵循论文的瓶颈设计"""

    def __init__(self):
        super(ResNet18Decoder, self).__init__()
        # 1. 逆转 final_project: (32, 7, 7) -> (256, 7, 7)
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS, 256, kernel_size=1, bias=False)

        # 2. [新] 瓶颈残差块 (模仿论文的 "Residual block x6")
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)  # 6个可能太多，先用4个
        )

        # 3. 逆转 layer3 (上采样): (256, 7, 7) -> (128, 14, 14)
        self.reverse_layer3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev3 = nn.BatchNorm2d(128)

        # 4. 逆转 layer2 (上采样): (128, 14, 14) -> (64, 28, 28)
        self.reverse_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev2 = nn.BatchNorm2d(64)

        # 5. 逆转 layer1 和 conv1 (细化)
        self.reverse_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)  # [新] 先处理
        x = F.relu(self.bn_rev3(self.reverse_layer3(x)))
        x = F.relu(self.bn_rev2(self.reverse_layer2(x)))
        x = self.reverse_conv1(x)
        return x


class VGG16Decoder(nn.Module):
    """对称于 VGG16, 严格遵循论文的瓶颈设计"""

    def __init__(self):
        super(VGG16Decoder, self).__init__()
        # 1. 逆转 final_project: (32, 7, 7) -> (256, 7, 7)
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS, 256, kernel_size=1, bias=False)

        # 2. [新] 瓶颈残差块
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )

        # 3. 逆转 VGG 的 C-C-C 块 (细化)
        self.reverse_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(True)
        )

        # 4. 逆转 Pool (上采样): (128, 7, 7) -> (128, 14, 14)
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        # 5. 逆转 C-C 块 (细化)
        self.reverse_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(True)
        )

        # 6. 逆转 Pool (上采样): (64, 14, 14) -> (64, 28, 28)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        # 7. 逆转 C-C 块 (细化) 和 最终输出
        self.reverse_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)  # [新] 先处理
        x = self.reverse_block3(x)
        x = F.relu(self.upsample1(x))
        x = self.reverse_block2(x)
        x = F.relu(self.upsample2(x))
        x = self.reverse_block1(x)
        return x


def get_decoder():
    """[新] 工厂函数：根据 config.py 返回对称的解码器"""
    if cfg.EXTRACTOR_TYPE == 'RESNET18_PRETRAINED':
        print("使用解码器: ResNet18Decoder (Symmetric, with Bottleneck Blocks)")
        return ResNet18Decoder()
    elif cfg.EXTRACTOR_TYPE == 'VGG16_PRETRAINED':
        print("使用解码器: VGG16Decoder (Symmetric, with Bottleneck Blocks)")
        return VGG16Decoder()
    elif cfg.EXTRACTOR_TYPE == 'SIMPLE_CNN':
        print("使用解码器: SimpleCNNDecoder (Symmetric, with Bottleneck Blocks)")
        return SimpleCNNDecoder()
    else:
        raise ValueError(f"未知的 EXTRACTOR_TYPE: {cfg.EXTRACTOR_TYPE}")


# ======================================================
# 2. Autoencoder & Data Loader
# ======================================================
class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = get_decoder()  # [新] 使用工厂函数

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction


def get_train_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root=cfg.DATA_ROOT, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


# ======================================================
# 3. Main Training Loop (Method 2: Reconstruct Original)
# ======================================================
def main():
    configure_model_from_path(RUN_DIR_TO_LOAD)

    print(f"正在从 {MODEL_PATH} 加载预训练的编码器...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。")
        sys.exit()

    checkpoint = torch.load(MODEL_PATH, map_location=cfg.DEVICE)
    if 'max_rules' not in checkpoint:
        print(f"错误: 检查点 '{MODEL_PATH}' 不包含 'max_rules'。")
        sys.exit()
    cfg.MAX_RULES = checkpoint['max_rules']
    print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")

    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    autoencoder = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)
    autoencoder.encoder.requires_grad_(False)
    print("编码器已冻结。开始训练解码器...")

    train_loader = get_train_loader()

    # [关键修改] 使用 MSELoss 来重建原始图像 (Method 2)
    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(DECODER_EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(cfg.DEVICE)

            optimizer.zero_grad()
            reconstructed_images = autoencoder(data)

            # [关键修改] 训练目标是原始图像 'data' (Method 2)
            loss = criterion(reconstructed_images, data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"[Epoch {epoch + 1}/{DECODER_EPOCHS}] Step {batch_idx + 1}/{len(train_loader)} | "
                      f"图像重建损失 (MSE): {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n==> Epoch {epoch + 1} 完成. 平均重建损失: {avg_loss:.6f}\n")

    torch.save(autoencoder.decoder.state_dict(), DECODER_SAVE_PATH)
    print(f"解码器训练完成！权重已保存至: {DECODER_SAVE_PATH}")


if __name__ == '__main__':
    main()