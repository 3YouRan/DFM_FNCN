import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# 导入我们的自定义模块
import config as cfg
from models import FullModel  # 需要 FullModel 的结构
from train_decoder import Decoder  # 需要 Decoder 的结构

# --- 配置 ---
MODEL_PATH = os.path.join('./checkpoints', 'best_model.pth')
DECODER_PATH = os.path.join('./checkpoints', 'decoder.pth')
OUTPUT_DIR = './checkpoints'

# [新] 输出文件名
GRID_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'rules_visualized_labeled.png')
INDIVIDUAL_IMAGES_PATH = os.path.join(OUTPUT_DIR, 'rule_images_labeled')

# [新] 图像和标签的布局设置
IMG_SCALE = 100  # 将 28x28 的图像放大到 100x100，以便看清
TEXT_HEIGHT = 35  # 为下方的标签文本留出 35 像素的高度
PADDING = 5  # 网格中图像之间的间距

# Fashion-MNIST 类别名称
CLASS_NAMES = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def main():
    print("开始可视化带标签的模糊规则...")

    # --- 1. 检查文件 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。请先运行 train.py。")
        return
    if not os.path.exists(DECODER_PATH):
        print(f"错误: 找不到解码器文件 '{DECODER_PATH}'。请先运行 train_decoder.py。")
        return

    if not os.path.exists(INDIVIDUAL_IMAGES_PATH):
        os.makedirs(INDIVIDUAL_IMAGES_PATH)

    # --- 2. 加载模型和解码器 ---
    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()
    print(f"已加载 DFM-FNCN 模型 (来自 {MODEL_PATH})")

    decoder = Decoder().to(cfg.DEVICE)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=cfg.DEVICE))
    decoder.eval()
    print(f"已加载解码器 (来自 {DECODER_PATH})")

    # --- 3. 提取规则信息 ---
    classifier = model.classifier
    num_active_rules = classifier.num_active_rules.item()
    if num_active_rules == 0:
        print("错误: 模型中没有激活的规则。")
        return

    print(f"发现 {num_active_rules} 条激活的规则。")

    # 提取规则的 Antecedents (中心)
    all_centers = classifier.centers.detach()
    active_centers = all_centers[:num_active_rules]

    # 提取规则的 Consequents (后件) 并找到每个规则的主要预测
    all_consequents = classifier.consequents.detach()
    active_consequents = all_consequents[:num_active_rules]
    consequents_softmax = F.softmax(active_consequents, dim=1)
    predicted_classes = torch.argmax(consequents_softmax, dim=1)

    # --- 4. 运行解码器 ---
    # 调整形状: [Num_Rules, N_Channels, 49] -> [Num_Rules, N_Channels, 7, 7]
    centers_reshaped = active_centers.view(
        num_active_rules, cfg.N_CHANNELS, cfg.IMG_DIM, cfg.IMG_DIM
    ).to(cfg.DEVICE)

    with torch.no_grad():
        visualized_rules = decoder(centers_reshaped).cpu()

    # 反归一化 (从 [-1, 1] -> [0, 1])
    visualized_rules = (visualized_rules + 1) / 2.0

    # --- 5. [新] 创建带标签的图像 ---
    labeled_tensors = []
    try:
        # 尝试加载默认字体；如果失败，Pillow 会使用一个内置的简单字体
        font = ImageFont.load_default()
    except IOError:
        print("警告: 无法加载默认字体，将使用 PIL 内置字体。")
        font = None

    print("正在为每条规则生成带标签的图像...")
    for i in range(num_active_rules):
        # 获取此规则的信息
        img_tensor = visualized_rules[i]
        pred_idx = predicted_classes[i].item()
        class_name = CLASS_NAMES[pred_idx]

        # 将 Tensor 转换为 PIL 图像 (用于绘制)
        pil_img = T.ToPILImage()(img_tensor)

        # 放大图像 (使用 NEAREST 保持像素感)
        pil_img_resized = pil_img.resize((IMG_SCALE, IMG_SCALE), resample=Image.Resampling.NEAREST)

        # 创建一个白底画布 (图像高度 + 文本区域高度)
        canvas = Image.new('RGB', (IMG_SCALE, IMG_SCALE + TEXT_HEIGHT), 'white')

        # 将放大后的图像粘贴到画布顶部
        canvas.paste(pil_img_resized, (0, 0))

        # 准备在画布上绘制文本
        draw = ImageDraw.Draw(canvas)

        # 绘制规则编号
        draw.text(
            (5, IMG_SCALE + 5),  # (x, y) 坐标
            f"Rule: {i}",  # 文本
            fill="black",  # 颜色
            font=font
        )

        # 绘制预测类别
        draw.text(
            (5, IMG_SCALE + 18),
            f"Pred: {class_name}",
            fill="blue",
            font=font
        )

        # 保存这张带标签的单图
        canvas.save(os.path.join(INDIVIDUAL_IMAGES_PATH, f"rule_{i}_({class_name}).png"))

        # 将带标签的 PIL 图像转换回 Tensor，以便存入网格
        labeled_tensors.append(T.ToTensor()(canvas))

    # --- 6. 保存最终的网格图 ---
    print("正在拼接最终的网格图...")
    # 自动计算网格的列数 (宽度)
    num_cols = int(np.ceil(np.sqrt(num_active_rules)))

    save_image(
        labeled_tensors,  # 使用我们带标签的图像列表
        GRID_IMAGE_PATH,
        nrow=num_cols,  # 设置网格宽度
        padding=PADDING,
        normalize=False  # 图像已经是 [0, 1] 范围
    )

    print("\n可视化完成!")
    print(f"带标签的网格图已保存至: {GRID_IMAGE_PATH}")
    print(f"带标签的单张图像已保存至: {INDIVIDUAL_IMAGES_PATH}/")


if __name__ == '__main__':
    main()