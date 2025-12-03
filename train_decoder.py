import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import config as cfg
from models import FullModel
import medmnist
from medmnist import BloodMNIST

# --- [重要] 在此处配置您要为其训练解码器的运行目录 ---
RUN_DIR_TO_LOAD = './checkpoints/FASHION_MNIST_DFM_FNCN_RESNET18_PRETRAINED_20251203_160957'
# ---

MODEL_PATH = os.path.join(RUN_DIR_TO_LOAD, 'best_model.pth')
DECODER_SAVE_PATH = os.path.join(RUN_DIR_TO_LOAD, 'decoder.pth')
DECODER_EPOCHS = 30
DECODER_LR = 0.001


def configure_model_from_checkpoint(checkpoint):
    """[新] 从检查点中的 'config_params' 推断配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    if params['MODEL_TYPE'] != 'DFM_FNCN':
        print("错误: 解码器只能为 DFM_FNCN 模型训练。")
        sys.exit()

    cfg.MODEL_TYPE = params['MODEL_TYPE']
    cfg.EXTRACTOR_TYPE = params['EXTRACTOR_TYPE']
    cfg.DATASET_NAME = params['DATASET_NAME']

    config_data = cfg.DATASET_CONFIGS[cfg.DATASET_NAME]
    cfg.N_CLASSES = config_data['n_classes']
    cfg.IN_CHANNELS = config_data['in_channels']
    cfg.CLASS_NAMES = config_data['class_names']
    cfg.TARGET_SIZE = config_data['target_size']

    print(f"推断配置: DATASET={cfg.DATASET_NAME}, MODEL={cfg.MODEL_TYPE}, EXTRACTOR={cfg.EXTRACTOR_TYPE}")


# ======================================================
# 1. 定义对称的解码器 (Decoders)
# ======================================================
class BasicBlock(nn.Module):
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
    def __init__(self):
        super(SimpleCNNDecoder, self).__init__()
        self.bottleneck = nn.Sequential(
            BasicBlock(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT),
            BasicBlock(cfg.N_CHANNELS_OUT, cfg.N_CHANNELS_OUT)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(cfg.N_CHANNELS_OUT, 16, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            # [新] 输出通道 = cfg.IN_CHANNELS
            nn.ConvTranspose2d(16, cfg.IN_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


class ResNet18Decoder(nn.Module):
    def __init__(self):
        super(ResNet18Decoder, self).__init__()
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS_OUT, 256, kernel_size=1, bias=False)
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256), BasicBlock(256, 256),
            BasicBlock(256, 256), BasicBlock(256, 256)
        )
        self.reverse_pool = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1)
        self.reverse_layer3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev3 = nn.BatchNorm2d(128)
        self.reverse_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_rev2 = nn.BatchNorm2d(64)
        self.reverse_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [新] 输出通道 = cfg.IN_CHANNELS
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)
        x = F.relu(self.reverse_pool(x))
        x = F.relu(self.bn_rev3(self.reverse_layer3(x)))
        x = F.relu(self.bn_rev2(self.reverse_layer2(x)))
        x = self.reverse_conv1(x)
        return x


class VGG16Decoder(nn.Module):
    def __init__(self):
        super(VGG16Decoder, self).__init__()
        self.reverse_project = nn.Conv2d(cfg.N_CHANNELS_OUT, 256, kernel_size=1, bias=False)
        self.bottleneck = nn.Sequential(
            BasicBlock(256, 256), BasicBlock(256, 256),
            BasicBlock(256, 256), BasicBlock(256, 256)
        )
        self.reverse_pool = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1)
        self.reverse_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.reverse_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(True)
        )
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.reverse_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            # [新] 输出通道 = cfg.IN_CHANNELS
            nn.Conv2d(64, cfg.IN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.reverse_project(x)
        x = self.bottleneck(x)
        x = F.relu(self.reverse_pool(x))
        x = self.reverse_block3(x)
        x = F.relu(self.upsample1(x))
        x = self.reverse_block2(x)
        x = F.relu(self.upsample2(x))
        x = self.reverse_block1(x)
        return x


def get_decoder():
    """工厂函数：根据 config.py 返回对称的解码器"""
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


class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = get_decoder()  # 使用工厂函数

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction


def get_train_loader():
    """[新] 根据 config.py 动态加载数据集"""
    if cfg.IN_CHANNELS == 1:
        norm_mean, norm_std = (0.5,), (0.5,)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    data_transform = transforms.Compose([
        transforms.Resize(cfg.TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    if cfg.DATASET_NAME == 'FASHION_MNIST':
        train_dataset = datasets.FashionMNIST(
            root=cfg.DATA_ROOT, train=True, download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'SVHN':
        train_dataset = datasets.SVHN(
            root=cfg.DATA_ROOT, split='train', download=True, transform=data_transform
        )
    elif cfg.DATASET_NAME == 'BLOOD_MNIST':
        train_dataset = BloodMNIST(
            split='train', transform=data_transform, download=True, root=cfg.DATA_ROOT
        )
    else:
        raise ValueError(f"未知的 DATASET_NAME: {cfg.DATASET_NAME}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


def main():
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.DEVICE)
    configure_model_from_checkpoint(checkpoint)

    print(f"正在从 {MODEL_PATH} 加载预训练的编码器...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。")
        sys.exit()

    cfg.MAX_RULES = checkpoint['max_rules']
    print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")

    base_model = FullModel().to(cfg.DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    autoencoder = Autoencoder(encoder=base_model.extractor).to(cfg.DEVICE)
    autoencoder.encoder.requires_grad_(False)
    print("编码器已冻结。开始训练解码器...")

    train_loader = get_train_loader()

    optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(DECODER_EPOCHS):
        total_loss = 0
        for batch_idx, data_tuple in enumerate(train_loader):
            # [新] 适配 MedMNIST 和 Torchvision
            data = data_tuple[0].to(cfg.DEVICE)

            optimizer.zero_grad()
            reconstructed_images = autoencoder(data)

            # 训练目标是原始图像 'data'
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