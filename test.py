import cv2
import numpy as np
import joblib
from skimage import measure
from skimage.feature import hog, local_binary_pattern
import os
import matplotlib.pyplot as plt


# 图像预处理
def preprocess_image(image, target_size=(100, 100)):
    resized_image = cv2.resize(image, target_size)  # 确保图像尺寸一致
    return resized_image


# 提取颜色直方图特征
def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 提取HOG特征
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


# 提取LBP特征
def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


# 特征提取函数
def extract_features(image):
    color_image = preprocess_image(image)
    color_histogram = extract_color_histogram(color_image)
    hog_features = extract_hog_features(color_image)
    lbp_features = extract_lbp_features(color_image)
    combined_features = np.hstack([color_histogram, hog_features, lbp_features])
    return combined_features


# 加载模型和缩放器
model_dir = 'E://Fruit-Images-Dataset-master//models'  # 模型保存路径
classifier = joblib.load(os.path.join(model_dir, 'fruit_classifier1.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler1.pkl'))
label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder1.pkl'))

# 确保分类器支持概率估计
if not hasattr(classifier, "predict_proba"):
    raise ValueError(
        "The classifier does not support probability estimates. Ensure the SVM classifier is trained with probability=True.")

# 读取图像并处理成灰度图像
image_path = 'fruits/test1.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 边缘检测
edges = cv2.Canny(gray, 50, 200)

# 形态学操作：膨胀和闭操作以连接边缘
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dilated = cv2.dilate(edges, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# 标记不同的图形
num_labels, labels = cv2.connectedComponents(closed)

# 舍弃面积较小的连通区域
min_area = 4000  # 设定一个阈值，舍弃面积小于该阈值的连通区域
for i in range(1, num_labels):
    if np.sum(labels == i) < min_area:
        labels[labels == i] = 0

# 重新标记
labels = measure.label(labels, connectivity=2)
num_labels = labels.max()


# 构建分类器并进行识别
def classify_fruit(image, label, classifier, scaler, label_encoder):
    mask = labels == label
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    fruit_image = image[y:y + h, x:x + w]

    # 确保分割后的图像调整为相同的大小
    fruit_image_resized = cv2.resize(fruit_image, (100, 100))
    features = extract_features(fruit_image_resized)
    features = scaler.transform([features])
    probabilities = classifier.predict_proba(features)[0]
    top_index = np.argmax(probabilities)
    fruit_label = label_encoder.inverse_transform([top_index])[0]
    probability = probabilities[top_index]
    print(f'Predicted class: {fruit_label}, Probability: {probability:.4f}')

    # 在原图上标记检测结果
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'{fruit_label}: {probability:.4f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


# 在原图上标记所有检测结果
for i in range(1, num_labels + 1):
    image = classify_fruit(image, i, classifier, scaler, label_encoder)

# 显示最终结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Fruits')
plt.show()
