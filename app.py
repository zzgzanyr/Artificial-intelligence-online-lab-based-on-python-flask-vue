import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.feature import hog, local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
import os
from glob import glob
import joblib

# 图像预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (100, 100))  # 调整大小以保证一致
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

# 数据增强（仅水平翻转）
def augment_image(image):
    images = [image]
    # 水平翻转
    images.append(cv2.flip(image, 1))
    return images

# 特征提取函数
def extract_features(image_path):
    color_image = preprocess_image(image_path)
    augmented_images = augment_image(color_image)
    combined_features = []
    for img in augmented_images:
        color_histogram = extract_color_histogram(img)
        hog_features = extract_hog_features(img)
        lbp_features = extract_lbp_features(img)
        combined = np.hstack([color_histogram, hog_features, lbp_features])
        combined_features.append(combined)
    return combined_features

# 加载数据集
def load_dataset(image_paths, labels):
    print(f"Starting to load dataset with {len(image_paths)} images...")
    features = []
    new_labels = []
    with ThreadPoolExecutor() as executor:
        all_features = list(executor.map(extract_features, image_paths))
    for i, f in enumerate(all_features):
        features.extend(f)
        new_labels.extend([labels[i]] * len(f))
    print("Finished loading dataset.")
    return np.array(features), np.array(new_labels)

# 训练SVM分类器
def train_classifier(features, labels):
    print("Starting training classifier...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    parameters = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    svc = svm.SVC(probability=True)
    classifier = GridSearchCV(svc, parameters, cv=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print("Finished training classifier.")
    return classifier, scaler

# 预测新图像
def predict_image(classifier, scaler, image_path):
    color_image = preprocess_image(image_path)
    color_histogram = extract_color_histogram(color_image)
    hog_features = extract_hog_features(color_image)
    lbp_features = extract_lbp_features(color_image)
    combined_features = np.hstack([color_histogram, hog_features, lbp_features])
    combined_features = scaler.transform([combined_features])
    probabilities = classifier.predict_proba(combined_features)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    return [(label_encoder.inverse_transform([i])[0], probabilities[i]) for i in top_indices]

# 示例使用
data_path = 'E://Fruit-Images-Dataset-master//Training'  # 根据实际路径修改

# 选择差异较大的类别
selected_categories = ['Apple Braeburn', 'Banana', 'Watermelon', 'Eggplant', 'Pineapple', 'Strawberry', 'Orange', 'Pear', 'Pomegranate']  # 根据需要选择差异较大的类别

model_dir = 'E://Fruit-Images-Dataset-master//models'
os.makedirs(model_dir, exist_ok=True)

# 合并所选类别的数据
all_image_paths = []
all_labels = []

for category in selected_categories:
    category_path = os.path.join(data_path, category)
    image_paths = glob(os.path.join(category_path, '*.jpg'))
    labels = [category] * len(image_paths)

    all_image_paths.extend(image_paths)
    all_labels.extend(labels)

    # 输出每个类别的进度信息
    print(f"Collected {len(image_paths)} images for category: {category}")

# 将标签转换为数字编码
label_encoder = LabelEncoder()
all_labels = label_encoder.fit_transform(all_labels)

# 加载数据集
features, all_labels = load_dataset(all_image_paths, all_labels)

# 训练分类器
classifier, scaler = train_classifier(features, all_labels)

# 保存模型和缩放器
joblib.dump(classifier, os.path.join(model_dir, 'fruit_classifier2.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler2.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder2.pkl'))
