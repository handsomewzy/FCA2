import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  # 需要安装 opencv-python 库

def load_rgb_images_as_hyperspectral(folder_path):
    """
    读取文件夹中的所有RGB图片，并将其按通道堆叠形成 (H, W, 3N) 维度的高光谱数据立方体
    :param folder_path: 图片文件夹路径
    :return: 拼接后的高光谱数据 (H, W, 3N)
    """
    image_list = []
    
    for filename in sorted(os.listdir(folder_path)):  # 按文件名排序，确保顺序
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # 读取彩色图片 (H, W, 3)
            if img is None:
                print(f"Warning: Failed to load {filename}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            image_list.append(img)

    if len(image_list) == 0:
        raise ValueError("No valid images found in the folder.")

    # 统一尺寸
    height, width, _ = image_list[0].shape
    image_stack = np.concatenate(image_list, axis=-1)  # 形成 (H, W, 3N)
    return image_stack

def plot_spectral_distribution(hypercube, output_file="video_spectral_distribution_rgb.png"):
    """
    绘制光谱分布曲线，每个通道分别计算均值
    :param hypercube: 高光谱数据 (H, W, 3N)
    :param output_file: 输出图片名称
    """
    num_bands = hypercube.shape[-1]  # 获取通道数 (3N)

    # 计算每个通道的平均值
    spectral_mean = np.mean(hypercube, axis=(0, 1))

    # 设置学术论文风格
    plt.rcParams.update({
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    # 生成不同颜色的曲线
    colors = ['red', 'green', 'blue'] * (num_bands // 3)  # RGB 通道循环

    # 绘制光谱通道分布曲线
    plt.figure(figsize=(10, 5), dpi=600)  # 高 DPI 输出
    # plt.plot(range(1, num_bands + 1), spectral_mean, 
    #          marker='o', linestyle='-', linewidth=1.5, markersize=3, 
    #          markeredgewidth=0.6, markeredgecolor='black', 
    #          label='Mean Spectral Response', color='black')
    
    plt.plot(range(1, num_bands + 1), spectral_mean, 
         marker='o', linestyle='-', linewidth=1.5, markersize=3, 
         markerfacecolor='red', markeredgewidth=0.6, markeredgecolor='black', 
         label='Mean Spectral Response')

    # 标注 RGB 通道分区
    for i in range(num_bands):
        plt.scatter(i + 1, spectral_mean[i], color=colors[i], s=10)  # 颜色区分 RGB
    
    # 设置轴标签
    plt.xlabel('Spectral Band Index (RGB Channels)', fontsize=14)
    plt.ylabel('Mean Reflectance', fontsize=14)
    plt.title('Spectral Response of Image Stack (RGB Channels)', fontsize=15)

    # 设置坐标刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 设置网格样式
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # 添加图例
    plt.legend(fontsize=12, loc='best', frameon=True, edgecolor='black')

    # 去除边框填充，符合学术论文标准
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()

    print(f"Spectral distribution plot saved as {output_file}")

# === 使用示例 ===
folder_path = "/data1/userhome/luwen/Code/wzy/Vid4/GT/calendar"  # 替换成你的图片文件夹路径
hypercube = load_rgb_images_as_hyperspectral(folder_path)
plot_spectral_distribution(hypercube)
