import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 读取 .mat 文件
mat_file = '/data1/userhome/luwen/Code/wzy/CAD2VSR/Indian_pines.mat'  # 确保此文件在当前目录
data = scipy.io.loadmat(mat_file)

# 假设数据存储在 'indian_pines' 键中（需要检查 .mat 文件的实际结构）
if 'indian_pines' in data:
    hyperspectral_data = data['indian_pines']
else:
    raise KeyError("Cannot find 'indian_pines' dataset. Check .mat file structure.")

# 获取数据维度
height, width, num_bands = hyperspectral_data.shape

# 计算每个光谱通道的均值
spectral_mean = np.mean(hyperspectral_data, axis=(0, 1))

# 设置学术论文风格（不使用 LaTeX）
plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# 绘制光谱通道分布曲线
plt.figure(figsize=(8, 5), dpi=600)  # 高 DPI 输出
plt.plot(range(1, num_bands + 1), spectral_mean, 
         marker='o', linestyle='-', linewidth=1.5, markersize=3, 
         markerfacecolor='red', markeredgewidth=0.6, markeredgecolor='black', 
         label='Mean Spectral Response')

# 设置轴标签和标题（不使用 LaTeX）
plt.xlabel('Spectral Band Index', fontsize=14)
plt.ylabel('Mean Reflectance', fontsize=14)
plt.title('Spectral Response of Indian Pines Dataset', fontsize=15)

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
output_image = "spectral_distribution_cvpr.png"
plt.savefig(output_image, dpi=600, bbox_inches='tight', transparent=True)
plt.show()

print(f"Spectral distribution plot saved as {output_image}")
