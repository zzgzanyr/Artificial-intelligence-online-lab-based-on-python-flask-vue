from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog, local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import joblib
from skimage import measure
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import base64

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
# 全局变量
fruit_classifier = None
fruit_scaler = None
fruit_label_encoder = None
model = None

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
    try:
        print(f"Extracting features from image with shape: {image.shape}")
        color_image = preprocess_image(image)
        print("Color image processed")
        color_histogram = extract_color_histogram(color_image)
        print("Color histogram extracted")
        hog_features = extract_hog_features(color_image)
        print("HOG features extracted")
        lbp_features = extract_lbp_features(color_image)
        print("LBP features extracted")
        combined_features = np.hstack([color_histogram, hog_features, lbp_features])
        print(f"Combined features length: {len(combined_features)}")
        return combined_features
    except Exception as e:
        print(f"Exception in extract_features: {str(e)}")
        raise e


# 特征提取函数
def extract_features_train(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read the image file from path: {image_path}")
    color_image = preprocess_image(image)
    color_histogram = extract_color_histogram(color_image)
    hog_features = extract_hog_features(color_image)
    lbp_features = extract_lbp_features(color_image)
    combined_features = np.hstack([color_histogram, hog_features, lbp_features])
    return combined_features
# 加载数据集
def load_dataset(image_paths, labels):
    print(f"Starting to load dataset with {len(image_paths)} images...")
    features = []
    new_labels = []
    with ThreadPoolExecutor() as executor:
        all_features = list(executor.map(extract_features_train, image_paths))
    for i, f in enumerate(all_features):
        features.append(f)
        new_labels.append(labels[i])
    print("Finished loading dataset.")
    return np.array(features), np.array(new_labels)

# 训练SVM分类器
def train_fruit_classifier(features, labels):
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

@app.route('/fruit/train', methods=['POST'])
def train_fruit():
    global fruit_classifier, fruit_scaler, fruit_label_encoder
    data = request.json
    data_path = data.get('data_path')
    selected_categories = data.get('categories')

    all_image_paths = []
    all_labels = []

    for category in selected_categories:
        category_path = os.path.join(data_path, category)
        image_paths = glob(os.path.join(category_path, '*.jpg'))
        labels = [category] * len(image_paths)

        all_image_paths.extend(image_paths)
        all_labels.extend(labels)

    fruit_label_encoder = LabelEncoder()
    all_labels = fruit_label_encoder.fit_transform(all_labels)

    features, all_labels = load_dataset(all_image_paths, all_labels)
    fruit_classifier, fruit_scaler = train_fruit_classifier(features, all_labels)

    return jsonify({'accuracy': accuracy_score(all_labels, fruit_classifier.predict(fruit_scaler.transform(features)))})


@app.route('/fruit/predict', methods=['POST'])
def predict_fruit():
    global fruit_classifier, fruit_scaler, fruit_label_encoder
    if not fruit_classifier or not fruit_scaler or not fruit_label_encoder:
        return jsonify({'error': 'Model is not loaded or trained.'})

    print("1: Received request to predict fruit")
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"2: Saved uploaded file to {filepath}")

    try:
        print("3: Attempting to read image")
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to read the image file from path: {filepath}")
        print("4: Image read successfully")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("5: Converted image to grayscale")

        edges = cv2.Canny(gray, 50, 200)
        print("6: Applied Canny edge detection")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        print("7: Performed morphological operations")

        num_labels, labels = cv2.connectedComponents(closed)
        print(f"8: Found {num_labels} connected components")

        min_area = 4000
        for i in range(1, num_labels):
            if np.sum(labels == i) < min_area:
                labels[labels == i] = 0
        print("9: Filtered small connected components")

        labels = measure.label(labels, connectivity=2)
        num_labels = labels.max()
        print(f"10: Relabeled connected components, now have {num_labels} components")

        predictions = []
        for i in range(1, num_labels + 1):
            print(f"11: Processing label {i}")
            mask = labels == i
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            print(f"12: Extracted bounding box: x={x}, y={y}, w={w}, h={h}")
            fruit_image = image[y:y + h, x:x + w]
            print(f"13: Extracted fruit image with shape {fruit_image.shape}")

            fruit_image_resized = cv2.resize(fruit_image, (100, 100))
            print(f"14: Resized fruit image to (100, 100)")

            features = extract_features(fruit_image_resized)
            print("15: Extracted features from resized fruit image")

            features = fruit_scaler.transform([features])
            probabilities = fruit_classifier.predict_proba(features)[0]
            top_index = np.argmax(probabilities)
            fruit_label = fruit_label_encoder.inverse_transform([top_index])[0]
            probability = probabilities[top_index]
            predictions.append((fruit_label, probability))

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{fruit_label}: {probability:.4f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            print(f"16: Labeled fruit: {fruit_label} with probability {probability:.4f}")

        os.remove(filepath)
        print(f"17: Removed temporary file {filepath}")

        _, img_encoded = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(img_encoded).decode('utf-8')
        print("18: Encoded image to base64")

        return jsonify({
            'image': base64_image,
            'predictions': predictions
        })

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({'error': str(e)})


@app.route('/fruit/upload_model', methods=['POST'])
def upload_fruit_model():
    global fruit_classifier
    print("Received request to upload fruit model")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        fruit_classifier = joblib.load(file_path)
        print(f"Model uploaded and loaded successfully from {file_path}")
        return jsonify({'message': 'Model uploaded and loaded successfully.'})
    else:
        print("Invalid file format")
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})

@app.route('/fruit/upload_scaler', methods=['POST'])
def upload_fruit_scaler():
    global fruit_scaler
    print("Received request to upload fruit scaler")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        fruit_scaler = joblib.load(file_path)
        print(f"Scaler uploaded and loaded successfully from {file_path}")
        return jsonify({'message': 'Scaler uploaded and loaded successfully.'})
    else:
        print("Invalid file format")
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})

@app.route('/fruit/upload_label_encoder', methods=['POST'])
def upload_fruit_label_encoder():
    global fruit_label_encoder
    print("Received request to upload fruit label encoder")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        fruit_label_encoder = joblib.load(file_path)
        print(f"Label encoder uploaded and loaded successfully from {file_path}")
        return jsonify({'message': 'Label Encoder uploaded and loaded successfully.'})
    else:
        print("Invalid file format")
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})


# 训练CNN模型
def train_model(conv_layers=2, filters=32, kernel_size=3, pool_size=2, dense_units=128, epochs=5):
    global model
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    model = Sequential()
    for _ in range(conv_layers):
        model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)
    return accuracy, history.history


@app.route('/train', methods=['POST'])
def train():
    params = request.json
    accuracy, history = train_model(**params)
    return jsonify({'accuracy': accuracy, 'history': history})


@app.route('/load', methods=['POST'])
def load_model_route():
    global model
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file.'})
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        if filepath.endswith('.h5'):
            model = load_model(filepath)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a .h5 model file.'})
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        os.remove(filepath)  # 删除上传的模型文件
        return jsonify({'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)})





@app.route('/save', methods=['POST'])
def save_model_route():
    global model
    path = request.json.get('path')
    print(path)
    try:
        model.save(path)
        return jsonify({'message': 'Model saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/validate', methods=['POST'])
def validate_model():
    global model
    if model is None:
        return jsonify({'error': 'Model is not loaded or trained.'})
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    try:
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img).argmax(axis=-1)[0]
        os.remove(filepath)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

# 图像预处理
def preprocess_image1(image_path, target_size=(100, 100)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.resize(image, target_size)  # 确保图像尺寸一致
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image, image

# 提取颜色直方图特征
def extract_color_histogram1(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 提取HOG特征
def extract_hog_features1(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# 特征提取函数
def extract_features1(image_path, target_size=(100, 100)):
    gray_image, color_image = preprocess_image1(image_path, target_size)
    color_histogram = extract_color_histogram1(color_image)
    hog_features = extract_hog_features1(gray_image)
    combined_features = np.hstack([color_histogram, hog_features])
    return combined_features

# 加载数据集
def load_dataset1(image_paths, labels):
    with ThreadPoolExecutor() as executor:
        features = list(executor.map(lambda p: extract_features1(p), image_paths))
    return np.array(features), np.array(labels)

# 训练SVM分类器
def train_classifier(features, labels):
    global scaler, label_encoder
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    classifier = svm.SVC(kernel='linear', probability=True)  # 启用概率估计
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return classifier, scaler, accuracy

@app.route('/svm/train', methods=['POST'])
def svm_train():
    global svm_classifier, scaler, label_encoder
    data_path = request.json.get('data_path')
    selected_categories = request.json.get('categories')

    all_image_paths = []
    all_labels = []

    for category in selected_categories:
        category_path = os.path.join(data_path, category)
        image_paths = glob(os.path.join(category_path, '*.jpg'))
        labels = [category] * len(image_paths)

        all_image_paths.extend(image_paths)
        all_labels.extend(labels)

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)

    features, all_labels = load_dataset1(all_image_paths, all_labels)
    svm_classifier, scaler, accuracy = train_classifier(features, all_labels)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(svm_classifier, os.path.join(model_dir, 'svm_classifier.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    return jsonify({'accuracy': accuracy})

@app.route('/svm/predict', methods=['POST'])
def svm_predict():
    global svm_classifier, scaler, label_encoder
    if not svm_classifier or not scaler or not label_encoder:
        return jsonify({'error': 'Model is not loaded or trained.'})

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    try:
        combined_features = extract_features1(filepath)
        combined_features = scaler.transform([combined_features])
        probabilities = svm_classifier.predict_proba(combined_features)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        predictions = [(label_encoder.inverse_transform([i])[0], probabilities[i]) for i in top_indices]
        os.remove(filepath)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

def save_svm_model(model, scaler, label_encoder, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path + '-svm.pkl')
    joblib.dump(scaler, save_path + '-scaler.pkl')
    joblib.dump(label_encoder, save_path + '-label_encoder.pkl')
    print("保存成功")
    return {'message': 'Model and related objects saved successfully.'}

@app.route('/save_svm', methods=['POST'])
def save_svm():
    global fruit_classifier, fruit_scaler, fruit_label_encoder
    print("正在保存")
    data = request.json
    save_path = data.get('save_path')
    print(save_path)
    if not save_path:
        return jsonify({'error': 'No save path provided.'}), 400

    if fruit_classifier is None or fruit_scaler is None or fruit_label_encoder is None:
        return jsonify({'error': 'SVM model, scaler, or label encoder not trained or loaded.'}), 400

    result = save_svm_model(fruit_classifier, fruit_scaler, fruit_label_encoder, save_path)
    return jsonify(result)

def load_global_model(file_path):
    global svm_classifier
    svm_classifier = joblib.load(file_path)

def load_global_scaler(file_path):
    global scaler
    scaler = joblib.load(file_path)

def load_global_label_encoder(file_path):
    global label_encoder
    label_encoder = joblib.load(file_path)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        load_global_model(file_path)
        return jsonify({'message': 'Model uploaded and loaded successfully.'})
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})

@app.route('/upload_scaler', methods=['POST'])
def upload_scaler():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        load_global_scaler(file_path)
        return jsonify({'message': 'Scaler uploaded and loaded successfully.'})
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})

@app.route('/upload_label_encoder', methods=['POST'])
def upload_label_encoder():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    if file and file.filename.endswith('.pkl'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        load_global_label_encoder(file_path)
        return jsonify({'message': 'Label Encoder uploaded and loaded successfully.'})
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .pkl file.'})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000, debug=True)

