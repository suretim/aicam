import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
#pip install Pillow

# 参数设置
output_dir = "data"
categories = ["healthy", "disease"]
img_size = (64, 64)
num_images = 100  # 每类生成100张

def generate_healthy_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))  # 健康绿色
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 20)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(2, 5)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(30, 130, 30))  # 模拟叶脉
    return img

def generate_disease_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))  # 同样叶色为底
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 10)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(5, 15)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(150, 75, 0))  # 棕色病斑
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img

# 创建目录
for category in categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# 生成图像
for category in tqdm(categories, desc="Generating images"):
    for i in range(num_images):
        if category == "healthy":
            img = generate_healthy_image()
        else:
            img = generate_disease_image()
        img.save(os.path.join(output_dir, category, f"{category}_{i}.png"))
