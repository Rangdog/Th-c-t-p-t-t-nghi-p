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
    print(f"Shape of preprocessed image: {img.shape}")
    return img


def augment_image(image):
    augmented_images = []

    # Ensure image is 3D (height, width, channel)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Check if the image is empty
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Empty image provided to augment_image function")
        return [image]  # Return the original image if it's empty

    # Convert back to uint8 for OpenCV operations
    image_uint8 = (image * 255).astype(np.uint8)

    # 1. Rotate image
    for angle in range(-30, 31, 10):
        M = cv2.getRotationMatrix2D(
            (image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(
            image_uint8, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated_image)

    # 2. Change brightness
    for brightness in [0.5, 1, 1.5]:
        bright_image = cv2.convertScaleAbs(
            image_uint8, alpha=brightness, beta=0)
        augmented_images.append(bright_image)

    # 3. Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image_uint8, noise)
    augmented_images.append(noisy_image)

    # 4. Crop image
    h, w = image.shape[:2]
    for _ in range(5):
        x = random.randint(0, w//4)
        y = random.randint(0, h//4)
        cropped_image = image_uint8[y:h-10, x:w-10]
        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            resized_image = cv2.resize(cropped_image, (img_width, img_height))
            augmented_images.append(resized_image)

    # 5. Flip horizontally
    flipped_image = cv2.flip(image_uint8, 1)
    augmented_images.append(flipped_image)

    # 6. Geometric transformation (Shear)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, random.randint(-10, 10)],
                   [0, 1, random.randint(-10, 10)]])
    sheared_image = cv2.warpAffine(image_uint8, M, (cols, rows))
    augmented_images.append(sheared_image)

    # Convert back to float32, normalize, and ensure 3D shape
    augmented_images = [img.astype(
        'float32') / 255.0 for img in augmented_images]
    augmented_images = [np.expand_dims(
        img, axis=-1) if len(img.shape) == 2 else img for img in augmented_images]

    for i, img in enumerate(augmented_images):
        print(f"Shape of augmented image {i}: {img.shape}")

    return augmented_images


# Hàm thêm dấu vân tay mới
def add_new_fingerprint(model, new_fingerprint_path, new_label):
    img = preprocess_images(new_fingerprint_path)
    print(f"Shape of new_image: {img.shape}")
    new_image = np.expand_dims(img, axis=0)
    print(f"Shape of new_image_expanded: {new_image.shape}")
    if new_image.shape[1:] != (img_height, img_width, 1):
        raise ValueError(
            f"Incorrect image shape: {new_image.shape[1:]}. Expected: ({img_height}, {img_width}, 1)")
    new_feature = model.predict(new_image)

    features = np.load('fingerprint_features.npy')
    labels = np.load('fingerprint_labels.npy')
    features = np.vstack([features, new_feature])
    labels = np.append(labels, new_label)

    # Augment the image
    augmented_images = augment_image(img)
    for i, aug_img in enumerate(augmented_images):
        print(f"Processing augmented image {i}")
        aug_img_expanded = np.expand_dims(aug_img, axis=0)
        print(
            f"Shape of augmented image before prediction: {aug_img_expanded.shape}")
        if aug_img_expanded.shape[1:] != (img_height, img_width, 1):
            print(
                f"Skipping augmented image with incorrect shape: {aug_img_expanded.shape}")
            continue
        aug_feature = model.predict(aug_img_expanded)
        features = np.vstack([features, aug_feature])
        labels = np.append(labels, new_label)
    np.save('fingerprint_features.npy', features)
    np.save('fingerprint_labels.npy', labels)


def get_new_label():
    try:
        labels = np.load("fingerprint_labels.npy")
        return len(labels)+1
    except Exception as e:
        return 0


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
        new_label = get_new_label()
        try:
            add_new_fingerprint(model, file_path, new_label)
            print("Fingerprint added successfully")
            return jsonify({'message': 'Fingerprint added successfully!'})
        except Exception as e:
            print(f"Error adding fingerprint: {str(e)}")
            return jsonify({'error': str(e)}), 500


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


def reset_fingerprint(feature_size):
    if os.path.exists("fingerprint_features.npy"):
        os.remove("fingerprint_features.npy")
    if os.path.exists("fingerprint_labels.npy"):
        os.remove("fingerprint_labels.npy")
    np.save('fingerprint_features.npy', np.empty((0, feature_size)))
    np.save('fingerprint_labels.npy', np.array([]))


@app.route('/api/reset', methods=['POST'])
def reset_fingerprint():
    try:
        output_shape = model.output_shape
        feature_size = output_shape[-1]
        reset_fingerprint(feature_size)
        return jsonify({"message": "Fingerprint data has been reset successfully."}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['VEFIFY_FOLDER']):
        os.makedirs(app.config['VEFIFY_FOLDER'])
    app.run(debug=True)
