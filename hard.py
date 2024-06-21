import cv2
import numpy as np
from skimage import measure, color
import matplotlib.pyplot as plt

# 读取图像并处理成灰度图像
image_path = 'fruits/hard1.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 边缘检测
edges = cv2.Canny(gray, 50, 180)

# 显示原始图像、灰度图像和边缘图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Gray')
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.show()

# 形态学操作：膨胀和闭操作以连接边缘
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilated = cv2.dilate(edges, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# 标记不同的图形
num_labels, labels = cv2.connectedComponents(closed)

# 舍弃面积较小的连通区域
min_area = 5000  # 设定一个阈值，舍弃面积小于该阈值的连通区域
for i in range(1, num_labels):
    if np.sum(labels == i) < min_area:
        labels[labels == i] = 0

# 重新标记
labels = measure.label(labels, connectivity=2)
num_labels = labels.max()

# 计算各个图形单元的周长
perimeters = np.array([np.sum(labels == i) for i in range(1, num_labels + 1)])

# 计算各个图形单元的面积
areas = np.array([np.sum(labels == i) for i in range(1, num_labels + 1)])

# 计算各个图形单元的圆度
circularities = 4 * np.pi * areas / (perimeters ** 2)

# 计算各个图像的颜色（色度）
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mean_hues = np.array([np.mean(hsv_image[labels == i][:, 0]) for i in range(1, num_labels + 1)])

# 显示检测到的边界和填充后的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges')
plt.subplot(1, 2, 2)
plt.imshow(closed, cmap='gray')
plt.title('Closed Result')
plt.show()

# 构建分类器并进行识别
def classify_fruit(mean_hue, area, circularity):
    if mean_hue > 0.5:
        return 'Peach'
    elif area == max(areas):
        return 'Pineapple'
    elif circularity < 0.5:
        return 'Banana'
    elif mean_hue < 0.125:
        return 'Pear'
    elif 1.0 < circularity < 1.25:
        return 'Apple'
    else:
        return 'Unknown'

for i in range(num_labels):
    label = i + 1
    fruit = classify_fruit(mean_hues[i], areas[i], circularities[i])
    mask = labels == label
    fruit_image = np.copy(image)
    fruit_image[~mask] = 0
    plt.figure()
    plt.imshow(cv2.cvtColor(fruit_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Fruit Category: {fruit}')
    plt.show()
