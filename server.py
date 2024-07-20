from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VEFIFY_FOLDER'] = 'verifys'
app.secret_key = 'supersecretkey'

img_height, img_width = 96, 96


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3)),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(600, activation='softmax')  # Lớp đầu ra cho phân loại
    ])
    return model


model = create_model()

model.load_weights('fingerprint_model_weights.weights.h5')


# Hàm tiền xử lý hình ảnh
def preprocess_images(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Thêm chiều kênh
    return img


def augment_image(image):
    augmented_images = []

    # 1. Xoay hình ảnh
    for angle in range(-30, 31, 10):  # Xoay từ -30 đến 30 độ với bước 10 độ
        M = cv2.getRotationMatrix2D(
            (image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated_image)

    # 2. Thay đổi độ sáng
    for brightness in [0.5, 1, 1.5]:  # Thay đổi độ sáng
        # beta=0 để không thay đổi độ tối
        bright_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        augmented_images.append(bright_image)

    # 3. Thêm nhiễu Gaussian
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    augmented_images.append(noisy_image)

    # 4. Cắt hình ảnh
    h, w = image.shape[:2]
    for _ in range(5):  # Tạo 5 hình ảnh cắt ngẫu nhiên
        x = random.randint(0, w//4)  # Cắt ngẫu nhiên từ 0 đến 1/4 chiều rộng
        y = random.randint(0, h//4)  # Cắt ngẫu nhiên từ 0 đến 1/4 chiều cao
        cropped_image = image[y:h-10, x:w-10]
        # Đảm bảo kích thước giống nhau
        augmented_images.append(cv2.resize(cropped_image, (128, 128)))

    # 5. Lật ngang
    flipped_image = cv2.flip(image, 1)  # Lật ngang
    augmented_images.append(flipped_image)

    # 6. Biến đổi hình học (Co giãn)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, random.randint(-10, 10)],
                   [0, 1, random.randint(-10, 10)]])  # Di chuyển ngẫu nhiên
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(sheared_image)

    return augmented_images


# Hàm thêm dấu vân tay mới
def add_new_fingerprint(model, new_fingerprint_path, new_label):
    augmented_images = preprocess_images(new_fingerprint_path)
    new_image = preprocess_images(new_fingerprint_path)
    new_image = np.expand_dims(new_image, axis=0)
    new_feature = model.predict(new_image)
    features = np.load('fingerprint_features.npy')
    labels = np.load('fingerprint_labels.npy')
    features = np.vstack([features, new_feature])
    labels = np.append(labels, new_label)
    for image in augmented_images:
        image = np.expand_dims(image, axis=0)
        # Dự đoán đặc trưng cho từng hình ảnh biến đổi
        new_feature = model.predict(image)
        features = np.vstack([features, new_feature])
        labels = np.append(labels, new_label)
    np.save('fingerprint_features.npy', features)
    np.save('fingerprint_labels.npy', labels)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        new_label = int(request.form['label'])
        add_new_fingerprint(model, file_path, new_label)
        return jsonify({'message': 'Fingerprint added successfully!'})


# Hàm xác thực dấu vân tay
def verify_fingerprint(model, fingerprint_image_path):
    input_image = preprocess_images(fingerprint_image_path)
    input_image = np.expand_dims(input_image, axis=0)
    input_feature = model.predict(input_image).flatten()
    features = np.load('fingerprint_features.npy')
    labels = np.load('fingerprint_labels.npy')
    similarities = cosine_similarity([input_feature], features)
    best_match_index = np.argmax(similarities)
    best_match_label = labels[best_match_index]
    return int(best_match_label), float(similarities[0][best_match_index])


@app.route('/api/verify', methods=['POST'])
def verify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['VEFIFY_FOLDER'], filename)
        file.save(file_path)
        best_match_label, similarity = verify_fingerprint(model, file_path)
        if similarity > 0.85:
            return jsonify({'message': 'Unlock successful!', 'label': best_match_label, 'similarities': similarity})
        else:
            return jsonify({'message': 'Unlock failed! No matching fingerprint found.'})


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
