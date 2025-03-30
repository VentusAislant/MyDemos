import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

model_path = "../../pretrained_models/openai/clip-vit-base-patch16"
# 加载 CLIP 模型（ViT-B/16）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPVisionModel.from_pretrained(model_path)
processor = CLIPImageProcessor.from_pretrained(model_path)

# 处理图像
image = Image.open("../../data/fish.jpg")  # 替换为你的图片路径
image_input = processor(image, return_tensors='pt')['pixel_values']

# 前向传播，获取注意力
with torch.no_grad():
    outputs = model(image_input, output_attentions=True)
    attentions = outputs.attentions  # shape: (num_layers, batch, num_heads, seq_len, seq_len)
    attentions = torch.stack(attentions)

print(attentions.shape)
attentions, _ = attentions.max(dim=2)  # (num_layers, batch, seq_len, seq_len)
print(attentions.shape)
# 取最后一层第一个头
attn_map = attentions[1][0].detach().cpu().numpy()  # shape=(seq_len, seq_len)

num_layers = attentions.shape[0]
seq_len = attentions.shape[-1]

# 选择 CLS Token（索引 0）关注的 Patch
cls_attns = attentions[:, 0, 0, 1:].reshape(num_layers, 14, 14)  # 14x14 Patch Grid

# 画图
fig, axes = plt.subplots(1, num_layers, figsize=(20, 4))

for i in range(num_layers):
    ax = axes[i]
    sns.heatmap(cls_attns[i].detach().cpu().numpy(), cmap="viridis", ax=ax)
    ax.set_title(f"Layer {i+1}")

plt.tight_layout()
plt.show()