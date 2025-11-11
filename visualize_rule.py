import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import sys

# 导入我们的自定义模块
import config as cfg
from models import FullModel
# [新] 从 train_decoder 导入所有可能的解码器和工厂函数
from train_decoder import BasicBlock, SimpleCNNDecoder, ResNet18Decoder, VGG16Decoder, get_decoder

# --- [重要] 在此处配置您要可视化的运行目录 ---
RUN_DIR_TO_VISUALIZE = './checkpoints/SVHN_DFM_FNCN_RESNET18_PRETRAINED_20251111_225408'
# ---

MODEL_PATH = os.path.join(RUN_DIR_TO_VISUALIZE, 'best_model.pth')
DECODER_PATH = os.path.join(RUN_DIR_TO_VISUALIZE, 'decoder.pth')
OUTPUT_DIR = RUN_DIR_TO_VISUALIZE

GRID_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'rules_visualized_labeled.png')
INDIVIDUAL_IMAGES_PATH = os.path.join(OUTPUT_DIR, 'rule_images_labeled')

IMG_SCALE = 100
TEXT_HEIGHT = 35
PADDING = 5


def configure_model_from_checkpoint(checkpoint):
    """[新] 从检查点中的 'config_params' 推断配置"""
    if 'config_params' not in checkpoint:
        print("错误: 这是一个旧的检查点。无法推断配置。")
        sys.exit()

    params = checkpoint['config_params']
    if params['MODEL_TYPE'] != 'DFM_FNCN':
        print("错误: 只有 DFM_FNCN 模型可以被可视化。")
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


def main():
    print("开始可视化带标签的模糊规则...")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DECODER_PATH):
        print(f"错误: 找不到 {MODEL_PATH} 或 {DECODER_PATH}。")
        print("请确保您已成功运行 train.py 和 train_decoder.py。")
        return
    if not os.path.exists(INDIVIDUAL_IMAGES_PATH):
        os.makedirs(INDIVIDUAL_IMAGES_PATH)

    # 1. 加载模型
    checkpoint = torch.load(MODEL_PATH, map_location=cfg.DEVICE)

    # 2. [新] 自动配置
    configure_model_from_checkpoint(checkpoint)

    cfg.MAX_RULES = checkpoint['max_rules']
    print(f"从检查点加载配置: MAX_RULES = {cfg.MAX_RULES}")

    model = FullModel().to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载 DFM-FNCN 模型 (来自 {MODEL_PATH})")

    # 3. [新] 加载正确的对称解码器
    decoder = get_decoder().to(cfg.DEVICE)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=cfg.DEVICE))
    decoder.eval()
    print(f"已加载对称解码器 (来自 {DECODER_PATH})")

    # 4. 提取规则信息
    classifier = model.classifier
    num_active_rules = classifier.num_active_rules.item()
    if num_active_rules == 0:
        print("错误: 模型中没有激活的规则。")
        return
    print(f"发现 {num_active_rules} 条激活的规则。")

    active_centers = classifier.centers.detach()[:num_active_rules]
    active_consequents = classifier.consequents.detach()[:num_active_rules]
    predicted_classes = torch.argmax(F.softmax(active_consequents, dim=1), dim=1)

    # 5. 运行解码器
    centers_reshaped = active_centers.view(
        num_active_rules, cfg.N_CHANNELS_OUT, cfg.IMG_DIM_OUT, cfg.IMG_DIM_OUT
    ).to(cfg.DEVICE)

    with torch.no_grad():
        visualized_rules = decoder(centers_reshaped).cpu()

    visualized_rules = (visualized_rules + 1) / 2.0

    # 6. 创建带标签的图像
    labeled_tensors = []
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    print("正在为每条规则生成带标签的图像...")
    for i in range(num_active_rules):
        img_tensor = visualized_rules[i]
        pred_idx = predicted_classes[i].item()
        class_name = cfg.CLASS_NAMES[pred_idx]

        # [新] 适配 3 通道图像
        if img_tensor.shape[0] == 1:
            pil_img = T.ToPILImage()(img_tensor)
        else:
            pil_img = T.ToPILImage(mode='RGB')(img_tensor)

        pil_img_resized = pil_img.resize((IMG_SCALE, IMG_SCALE), resample=Image.Resampling.NEAREST)

        # [新] 确保画布是 RGB 以便粘贴
        canvas = Image.new('RGB', (IMG_SCALE, IMG_SCALE + TEXT_HEIGHT), 'white')
        canvas.paste(pil_img_resized, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((5, IMG_SCALE + 5), f"Rule: {i}", fill="black", font=font)
        draw.text((5, IMG_SCALE + 18), f"Pred: {class_name}", fill="blue", font=font)

        canvas.save(os.path.join(INDIVIDUAL_IMAGES_PATH, f"rule_{i}_({class_name}).png"))
        labeled_tensors.append(T.ToTensor()(canvas))

    # 7. 保存最终的网格图
    print("正在拼接最终的网格图...")
    num_cols = int(np.ceil(np.sqrt(num_active_rules)))
    if num_cols == 0:
        print("没有规则可供可视化。")
        return

    save_image(
        labeled_tensors,
        GRID_IMAGE_PATH,
        nrow=num_cols,
        padding=PADDING,
        normalize=False
    )

    print("\n可视化完成!")
    print(f"带标签的网格图已保存至: {GRID_IMAGE_PATH}")
    print(f"带标签的单张图像已保存至: {INDIVIDUAL_IMAGES_PATH}/")


if __name__ == '__main__':
    main()