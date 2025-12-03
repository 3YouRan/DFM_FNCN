# **DFM-FNCN: 视觉可解释的模糊神经网络 (PyTorch 复现)**

本项目是对论文 **"Visually Interpretable Fuzzy Neural Classification Network With Deep Convolutional Feature Maps" (DFM-FNCN)** 核心思想的 PyTorch 实现。

本项目不仅仅是一个简单的复现，而是一个**可配置的实验框架**，允许您：

1. 训练论文中提出的 **DFM-FNCN**（一种具有动态模糊规则的可解释模型）。  
2. 训练一个**传统的 DCNN**（使用相同的主干网络 \+ 全连接层）作为性能基准进行对比。  
3. 在多个不同的特征提取器（ResNet-18, VGG-16, SimpleCNN）之间轻松切换。  
4. 完整复现论文的**视觉可解释性**流程：训练一个对称的解码器，并将学到的模糊规则可视化为清晰的图像。

\<\!-- 建议：替换为您生成的 rules\_visualized\_labeled.png 的截图 \--\>

## **核心特性**

* **模型对比框架**:  
  * **DFM\_FNCN (论文模型)**: 实现了论文中的动态在线规则生成（基于 PHI\_TH 阈值）。  
  * **TRADITIONAL\_CNN (基准模型)**: 使用与模糊模型完全相同的主干网络，后跟你可以在 config.py 中自定义的全连接层。  
* **可插拔的特征提取器**:  
  * RESNET18\_PRETRAINED: 基于 ImageNet 预训练的 ResNet-18。  
  * VGG16\_PRETRAINED: 基于 ImageNet 预训练的 VGG-16。  
  * SIMPLE\_CNN: 一个用于快速测试的轻量级 CNN。  
* **严格的论文复现 (数值稳定)**:  
  * **torch.prod 连乘**: 严格遵循论文公式 (7)（激发强度为隶属度的连乘）。  
  * **64位精度修复**: 为了解决 N=32 时 float32 连乘导致的**数值下溢**（模式崩溃）问题，关键的 phi 计算步骤已升级到 float64 (双精度) 进行，确保梯度回传，模型可以正常训练。  
  * **动态状态保存**: num\_active\_rules（激活的规则数）被正确注册为 PyTorch 的**缓冲区 (buffer)**，确保它能随模型权重 (best\_model.pth) 一起保存和加载。  
* **完整的可解释性流程**:  
  * **train\_decoder.py**: 实现了论文 **Section VII (Fig. 15\)** 中讨论的第二种（免标注）训练方法。  
  * **对称解码器**: 脚本会根据您选择的编码器（ResNet/VGG/SimpleCNN）**自动构建一个对称的解码器**（包含论文中提到的**瓶颈残差块**），以确保高质量的图像重建。  
  * **visualize\_rule.py**: 加载您训练好的模型和解码器，将抽象的模糊规则中心（centers）解码为带标签的、可理解的图像。

## **项目文件结构**

.  
├── checkpoints/                \# 训练运行的输出目录  
│   └── DFM\_FNCN\_RESNET18.../   \# 每个运行的唯一目录  
│       ├── best\_model.pth      \# 保存的最佳模型权重  
│       ├── decoder.pth         \# 训练好的解码器  
│       ├── training\_history.png    \# 训练历史图表  
│       ├── fuzzy\_rules\_consequents.png \# 规则后件热图  
│       ├── classification\_report.csv \# 推理报告  
│       ├── confusion\_matrix.png    \# 推理混淆矩阵  
│       └── rules\_visualized\_labeled.png \# 最终的可视化剪影  
│  
├── config.py                   \# \[控制中心\] 在此配置实验  
├── models.py                   \# 包含所有模型架构 (3个提取器, 2个分类器)  
├── train.py                    \# \[步骤1\] 运行此文件来训练模型  
├── validation.py               \# \[步骤2\] 运行此文件来评估模型  
├── train\_decoder.py            \# \[步骤3\] 运行此文件来训练解码器  
└── visualize\_rule.py           \# \[步骤4\] 运行此文件来可视化规则

## **依赖库**

您需要安装以下 Python 库：

pip install torch torchvision  
pip install numpy  
pip install pandas  
pip install scikit-learn  
pip install matplotlib  
pip install seaborn  
pip install Pillow

## **完整工作流 (如何使用)**

这是一个四步流程，用于完整地复现论文的分类和可视化结果。

### **步骤 1: 训练模型 (train.py)**

1. **配置**: 打开 config.py。  
2. 设置 MODEL\_TYPE: 选择 'DFM\_FNCN'（论文模型）或 'TRADITIONAL\_CNN'（基准模型）。  
3. 设置 EXTRACTOR\_TYPE: 选择 'RESNET18\_PRETRAINED', 'VGG16\_PRETRAINED' 或 'SIMPLE\_CNN'。  
4. **训练**: 运行训练脚本。  
   ```python train.py```

5. 输出: 脚本会自动在 checkpoints/ 目录下创建一个唯一的、带时间戳的目录，例如 checkpoints/DFM\_FNCN\_RESNET18\_PRETRAINED\_20251111\_1830。  
   所有训练结果，包括 best\_model.pth，都会保存在这个新目录中。

### **步骤 2: 评估模型 (validation.py)**

1. **配置**: 打开 validation.py。  
2. 将顶部的 RUN\_DIR\_TO\_VALIDATE 变量设置为您在步骤 1 中创建的**运行目录路径**。  
   \# 示例:  
   RUN\_DIR\_TO\_VALIDATE \= './checkpoints/DFM\_FNCN\_RESNET18\_PRETRAINED\_20251111\_1830'

3. **运行**:  
   ```python validation.py```

4. 输出: 脚本会自动从目录名中推断模型配置，加载模型，并在控制台打印详细的分类报告。  
   classification\_report.csv 和 confusion\_matrix.png 将被保存到您的运行目录中。

### **步骤 3: 训练解码器 (train\_decoder.py)**

此步骤**仅**适用于 DFM\_FNCN 模型。

1. **配置**: 打开 train\_decoder.py。  
2. 将顶部的 RUN\_DIR\_TO\_LOAD 变量设置为您要可视化的 DFM\_FNCN **运行目录路径**。  
   \# 示例:  
   ```RUN\_DIR\_TO\_LOAD \= './checkpoints/DFM\_FNCN\_RESNET18\_PRETRAINED\_20251111\_1830'```

3. **运行**:  
   ```python train\_decoder.py```

4. **输出**: 脚本会加载 best\_model.pth，提取其编码器，训练一个对称的解码器，并将 decoder.pth 保存回**相同**的运行目录中。

### **步骤 4: 可视化模糊规则 (visualize\_rule.py)**

此步骤**仅**适用于 DFM\_FNCN 模型，且**必须**在步骤 3 之后运行。

1. **配置**: 打开 visualize\_rule.py。  
2. 将顶部的 RUN\_DIR\_TO\_VISUALIZE 变量设置为您的**运行目录路径**。  
   \# 示例:  
   ```RUN\_DIR\_TO\_VISUALIZE \= './checkpoints/DFM\_FNCN\_RESNET18\_PRETRAINED\_20251111\_1830'```
3. **运行**:  
   ```python visualize\_rule.py```

4. 输出: 脚本会加载 best\_model.pth（用于获取规则）和 decoder.pth（用于绘图）。  
   最终的可视化结果 rules\_visualized\_labeled.png（带标签的剪影网格）将保存到您的运行目录中。

# 下一步计划
目前模型在SVHM和Fashion-mnist上的可视化的效果不太理想，下一步计划尝试更多的数据集（如GTSRB的子集）；模型对比增加其他深度模糊系统

## 可能实现的创新点1 - 模糊层改进
1. **新参数**：在 Dynamic\_DFM\_FNCN 类中，定义一个新的可学习参数 alpha (注意力权重)，其形状为 (MAX\_RULES, N\_CHANNELS\_OUT)。  
2. **修改公式 (7)**：将激发强度 phi 的计算从“硬性”连乘改为“加权”聚合。  
   * **方案 (稳定)**：使用加权平均代替连乘：phi \= torch.sum(att\_weights \* mu, dim=2)  
   * 其中 att\_weights \= F.softmax(self.alpha\[:active\_rules\_count\], dim=1)  
3. **训练过程**：alpha 参数将通过标准的反向传播自动学习。
## 可能实现的创新点2 - 结构学习方法改进
在训练过程中或训练后，**自动修剪 (Prune)** 掉无效的规则。
1. **评估指标**：为每条规则计算一个“重要性得分”。  
2. **修剪标准（二选一）**：  
   * **标准 A：基于后件 (Consequent-based)**：在 train.py 的 visualize\_and\_save\_rules 中，我们已经计算了 consequents\_softmax。如果一条规则的“置信度”最高（即 softmax 的最大值）都低于一个阈值（例如 0.3），说明这条规则“不确定”它自己属于哪个类，应被修剪。  
   * **标准 B：基于激活 (Activation-based)**：在 evaluate 函数中，跟踪每条规则的平均激发强度 phi\_norm。如果一条规则在整个测试集上的平均激活度都接近于 0，说明它是一条“死规则”，应被修剪。  
3. **实现**：训练完成后，运行一个 prune\_model() 函数，它会识别出“坏”规则，然后将“好”规则的 centers, widths, consequents 参数复制到一个新的、更小的模型中，并保存这个“紧凑版”模型。
## 可能实现的创新点3 - 结构学习方法改进
**1\. 现有局限性 (The Problem)**

目前的规则生成机制是**路径依赖的 (Path-dependent)**。

* 规则是基于**单个样本**（x\_flat\[min\_idx\]）创建的。  
* 如果模型“运气不好”，在训练初期为“Ankle boot”选择了一个非常糟糕、有噪声的样本作为初始中心，这条规则可能需要很长时间来“纠正”自己，甚至可能永远无法成为一个好的原型。

**2\. 创新方案 (The Solution)**

将“在线”生成规则改为“批量”初始化，使用**聚类算法**来找到最佳的初始规则中心。

1. **预计算**：在 train.py 开始训练**之前**，先用 model.extractor 完整地遍历一次**整个训练集**，收集所有样本的特征图 (60000, 128, 36)。  
2. **聚类**：对这些特征图运行一个聚类算法（例如 K-Means，或者更适合的 **Fuzzy C-Means**），以 N\_CLASSES（例如 10）或稍大的 K（例如 20）为聚类中心数。  
3. **初始化**：使用这 K 个聚类中心作为 Dynamic\_DFM\_FNCN 的**初始规则中心 (centers)**。  
4. **训练**：关闭动态规则生成（PHI\_TH 设为 0），只对这 K 条规则的参数（centers, widths, consequents）进行微调。