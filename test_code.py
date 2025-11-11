import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. 超参数与设置 ---
BATCH_SIZE = 64
EPOCHS = 5  # 为了演示快速运行，我先设为5，您可以改回10或更多
LR = 0.002  # 稍微调高了一点学习率以便更快收敛
N_RULES = 20  # 增加一些规则数以提高容量
N_CLASSES = 10  # Fashion-MNIST 有10个类别
N_CHANNELS = 32  # 我们的CNN特征提取器输出的通道数
IMG_DIM = 7  # 我们的CNN特征提取器输出的特征图空间维度 (7x7)
P_DIM = IMG_DIM * IMG_DIM  # p = w * h (论文中的 p = w x h = 49)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 数据加载 (Fashion-MNIST) ---
# [修复部分]：加入了具体的预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将图像像素值归一化到 [-1, 1]
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)  # 测试集batch调大点加快速度


# --- 3. DCNN 特征提取器 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 16x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, N_CHANNELS, kernel_size=3, padding=1),  # -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 32x7x7
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# --- 4. DFM-FNCN 分类器核心实现 ---
class DFM_FNCN(nn.Module):
    def __init__(self, n_channels, p_dim, n_classes, n_rules):
        super(DFM_FNCN, self).__init__()
        self.n_channels = n_channels
        self.p_dim = p_dim
        self.n_classes = n_classes
        self.n_rules = n_rules

        # 1. 特征图批量归一化
        self.bn = nn.BatchNorm2d(n_channels)

        # 2. 可学习参数
        # 前件中心 C_j^i: (Rules, Channels, P_dim)
        self.centers = nn.Parameter(torch.randn(n_rules, n_channels, p_dim) * 0.01)
        # 前件宽度 Sigma_j^i: (Rules, Channels)
        self.widths_param = nn.Parameter(torch.ones(n_rules, n_channels) * 0.5)
        # 后件权重 W_s^i: (Rules, Classes)
        self.consequents = nn.Parameter(torch.randn(n_rules, n_classes) * 0.01)

    def forward(self, x):
        # x: (Batch, Channels, Height, Width) = (B, 32, 7, 7)
        b = x.size(0)

        # --- 预处理 ---
        x = self.bn(x)
        # 展平: (B, 32, 49)
        x_flat = x.view(b, self.n_channels, -1)

        # --- 模糊化 (Fuzzification) ---
        # 扩展维度以进行广播计算
        x_exp = x_flat.unsqueeze(1)  # (B, 1, 32, 49)
        c_exp = self.centers.unsqueeze(0)  # (1, R, 32, 49)

        # 计算匹配度 (Matching Degree): 余弦相似度
        # 沿着 P_DIM (dim=3) 计算 -> (B, R, 32)
        M = F.cosine_similarity(x_exp, c_exp, dim=3)

        # 计算距离 d = 1 - M
        d = 1.0 - M

        # 计算隶属度 mu = exp(-d^2 / sigma^2)
        # 使用 softplus 保证宽度为正数
        sigma = F.softplus(self.widths_param).unsqueeze(0) + 1e-6  # (1, R, 32)
        mu = torch.exp(-torch.pow(d, 2) / torch.pow(sigma, 2))  # (B, R, 32)

        # --- 规则激发 (Firing Strength) ---
        # 使用乘积算子 (Product T-norm) 聚合所有通道的隶属度
        # phi: (B, R)
        phi = torch.prod(mu, dim=2)

        # --- 去模糊化 (Defuzzification) ---
        # 归一化激发强度
        phi_sum = torch.sum(phi, dim=1, keepdim=True) + 1e-9
        phi_norm = phi / phi_sum  # (B, R)

        # 加权平均计算输出 logits
        # (B, R) @ (R, Classes) -> (B, Classes)
        output_logits = torch.matmul(phi_norm, self.consequents)

        return output_logits


# --- 5. 完整模型 ---
class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.extractor = SimpleCNN()
        self.classifier = DFM_FNCN(N_CHANNELS, P_DIM, N_CLASSES, N_RULES)

    def forward(self, x):
        features = self.extractor(x)
        logits = self.classifier(features)
        return logits


# --- 6. 训练主循环 ---
if __name__ == '__main__':
    model = FullModel().to(device)
    # 使用 CrossEntropyLoss (包含 Softmax 和 NLLLoss)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"[Epoch {epoch + 1}/{EPOCHS}] Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100. * correct_train / total_train:.2f}%")

        # --- 每个 Epoch 结束后进行测试 ---
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()

        print(f"==> Epoch {epoch + 1} 完成. 测试集准确率: {100. * correct_test / total_test:.2f}%\n")

    print("训练结束.")