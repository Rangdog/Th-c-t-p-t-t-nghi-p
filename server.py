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
import time
import matplotlib.pyplot as plt

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
model2 = create_model()

model.load_weights('fingerprint_model_weights1.weights.h5')
model2.load_weights('fingerprint_model_weights2.weights.h5')

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

    # Đảm bảo hình ảnh có 3 chiều (chiều cao, chiều rộng, kênh)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Kiểm tra xem hình ảnh có rỗng không
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Empty image provided to augment_image function")
        return [image]  # Trả về hình ảnh gốc nếu rỗng

    # Chuyển đổi về uint8 cho các thao tác OpenCV
    image_uint8 = (image * 255).astype(np.uint8)

    # 1. Xoay hình ảnh với padding
    for angle in range(-30, 31, 10):
        # Tạo một canvas trắng
        padded_image = cv2.copyMakeBorder(
            image_uint8, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        M = cv2.getRotationMatrix2D(
            (padded_image.shape[1] // 2, padded_image.shape[0] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(
            padded_image, M, (padded_image.shape[1], padded_image.shape[0]))
        # Crop lại kích thước gốc
        cropped_image = rotated_image[20:-20, 20:-20]
        augmented_images.append(cropped_image)

    # 2. Thay đổi độ sáng
    for alpha in [0.5, 1.0, 1.5]:  # Tăng/giảm độ sáng
        bright_image = cv2.convertScaleAbs(image_uint8, alpha=alpha, beta=0)
        augmented_images.append(bright_image)

    # 3. Thêm nhiễu Gaussian
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image_uint8, noise)
    augmented_images.append(noisy_image)

    # 4. Làm mờ Gaussian
    blurred_image = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    augmented_images.append(blurred_image)

    # 5. Biến hình đàn hồi
    def elastic_transform(image, alpha, sigma):
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = random_state.normal(0, sigma, size=shape[0]).astype(np.float32)
        dy = random_state.normal(0, sigma, size=shape[1]).astype(np.float32)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        distorted_image = cv2.remap(image, (x + dx[:, np.newaxis]).astype(np.float32),
                                    (y + dy[np.newaxis, :]).astype(np.float32),
                                    interpolation=cv2.INTER_LINEAR)
        return distorted_image

    elastic_image = elastic_transform(image_uint8, alpha=34, sigma=4)
    augmented_images.append(elastic_image)

    # 6. Cắt ngẫu nhiên
    h, w = image.shape[:2]
    for _ in range(5):
        # Tạo cắt ngẫu nhiên trong khoảng 1/4 kích thước
        x = random.randint(0, w // 4)
        y = random.randint(0, h // 4)
        cropped_image = image_uint8[y:h-10, x:w-10]
        if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            # Đảm bảo kích thước là 96x96
            resized_image = cv2.resize(cropped_image, (96, 96))
            # Tạo nền trắng
            white_background = np.ones((96, 96, 3), dtype=np.uint8) * 255
            if resized_image.ndim == 2:  # Nếu là hình ảnh grayscale, chuyển sang RGB
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            # Chèn hình ảnh vào nền trắng
            white_background[:resized_image.shape[0],
                             :resized_image.shape[1]] = resized_image
            augmented_images.append(white_background)

    # 7. Lật hình ảnh theo chiều ngang
    flipped_image = cv2.flip(image_uint8, 1)
    white_flipped_background = np.ones(
        (96, 96, 3), dtype=np.uint8) * 255  # Nền trắng
    if flipped_image.ndim == 2:  # Nếu là hình ảnh grayscale
        flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_GRAY2BGR)
    white_flipped_background[:flipped_image.shape[0],
                             :flipped_image.shape[1]] = flipped_image
    augmented_images.append(white_flipped_background)

    # 10. Thêm nhiễu muối và tiêu
    def salt_and_pepper_noise(image, salt_prob, pepper_prob):
        # Kiểm tra xem hình ảnh có rỗng không
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            print("Error: Input image is empty or has invalid dimensions.")
            return image  # Trả về hình ảnh gốc nếu rỗng

        noisy = np.copy(image)

        # Thêm nhiễu muối
        num_salt = np.ceil(salt_prob * image.size).astype(int)  # Số lượng muối
        if num_salt > 0:  # Kiểm tra nếu có muối để thêm
            coords = [np.random.randint(0, i, num_salt) for i in image.shape]
            noisy[coords[0], coords[1]] = 1  # Thiết lập nhiễu muối thành trắng

        # Thêm nhiễu tiêu
        num_pepper = np.ceil(
            pepper_prob * image.size).astype(int)  # Số lượng tiêu
        if num_pepper > 0:  # Kiểm tra nếu có tiêu để thêm
            coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
            noisy[coords[0], coords[1]] = 0  # Thiết lập nhiễu tiêu thành đen

        return noisy

    # Thêm nhiễu muối và tiêu
    sp_image = salt_and_pepper_noise(
        image_uint8, salt_prob=0.02, pepper_prob=0.02)
    white_sp_background = np.ones(
        (96, 96, 3), dtype=np.uint8) * 255  # Nền trắng
    if sp_image.ndim == 2:  # Nếu là hình ảnh grayscale
        sp_image = cv2.cvtColor(sp_image, cv2.COLOR_GRAY2BGR)
    white_sp_background[:sp_image.shape[0], :sp_image.shape[1]] = sp_image
    augmented_images.append(white_sp_background)

    # Chuyển đổi lại về float32, chuẩn hóa, và đảm bảo hình dạng 3D
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
    # # Kiểm tra số lượng hình ảnh tăng cường
    # num_images_to_plot = min(22, len(augmented_images)
    #                          )  # Vẽ tối đa 22 hình ảnh

    # # Tạo hình vẽ với kích thước lớn hơn để chứa tất cả hình ảnh
    # fig, axes = plt.subplots(4, 6, figsize=(20, 15))  # Tạo lưới 4 hàng, 6 cột

    # for i in range(num_images_to_plot):
    #     aug_img = augmented_images[i]

    #     # Hiển thị hình ảnh
    #     # Sử dụng squeeze để bỏ chiều kênh nếu cần
    #     axes[i // 6, i % 6].imshow(aug_img.squeeze(), cmap='gray')
    #     axes[i // 6, i % 6].axis('off')  # Tắt trục
    #     axes[i // 6, i % 6].set_title(f'Augmented Image {i + 1}')  # Tiêu đề

    # # Tắt trục không sử dụng (nếu có)
    # for j in range(num_images_to_plot, axes.size):
    #     axes[j // 6, j % 6].axis('off')

    # # Điều chỉnh lưới để không bị chồng chéo
    # plt.tight_layout()
    # plt.show()  # Hiển thị hình ảnh
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
# Hàm thêm dấu vân tay mới


def add_new_fingerprint_2(model, new_fingerprint_path, new_label):
    img = preprocess_images(new_fingerprint_path)
    print(f"Shape of new_image: {img.shape}")
    new_image = np.expand_dims(img, axis=0)
    print(f"Shape of new_image_expanded: {new_image.shape}")
    if new_image.shape[1:] != (img_height, img_width, 1):
        raise ValueError(
            f"Incorrect image shape: {new_image.shape[1:]}. Expected: ({img_height}, {img_width}, 1)")
    new_feature = model.predict(new_image)

    features = np.load('fingerprint_features1.npy')
    labels = np.load('fingerprint_labels1.npy')
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
    np.save('fingerprint_features1.npy', features)
    np.save('fingerprint_labels1.npy', labels)


def get_new_label():
    try:
        labels = np.load("fingerprint_labels.npy")
        # Lấy các giá trị duy nhất từ labels
        unique_labels = np.unique(labels)

        # Trả về số lượng nhãn duy nhất
        return len(unique_labels) + 1
    except Exception as e:
        return 0


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    print(file.name)
    # Kiểm tra xem file có phải là hình ảnh không
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File is not an image'}), 400

    # Lưu tệp vào thư mục đã chỉ định
    if file:
        # Tạo tên tệp duy nhất
        filename = f"{int(time.time())}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Lưu tệp
        file.save(file_path)

        new_label = get_new_label()  # Lấy new_label

        try:
            add_new_fingerprint(model, file_path, new_label)
            add_new_fingerprint_2(model2, file_path, new_label)
            print("Fingerprint added successfully")
            # Trả về new_label trong phản hồi
            return jsonify({'message': 'Fingerprint added successfully!', 'new_label': new_label}), 200
        except Exception as e:
            print(f"Error adding fingerprint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500


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

# Hàm xác thực dấu vân tay


def verify_fingerprint_2(model, fingerprint_image_path):
    input_image = preprocess_images(fingerprint_image_path)
    input_image = np.expand_dims(input_image, axis=0)
    input_feature = model.predict(input_image).flatten()
    features = np.load('fingerprint_features1.npy')
    labels = np.load('fingerprint_labels1.npy')
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
        print("label: ", best_match_label)
        print("similarity: ", similarity)
        if similarity > 0.95:
            return jsonify({'message': 'Unlock successful!', 'label': best_match_label, 'similarities': similarity})
        else:
            return jsonify({'message': 'Unlock failed! No matching fingerprint found.', 'label': best_match_label, 'similarities': similarity})


@app.route('/api/verify_model_2', methods=['POST'])
def verify_image_by_model_2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['VEFIFY_FOLDER'], filename)
        file.save(file_path)
        best_match_label, similarity = verify_fingerprint_2(model2, file_path)
        print("label by model2: ", best_match_label)
        print("similarity by model2: ", similarity)
        if similarity > 0.99:
            return jsonify({'message': 'Unlock successful!', 'label': best_match_label, 'similarities': similarity})
        else:
            return jsonify({'message': 'Unlock failed! No matching fingerprint found.', 'label': best_match_label, 'similarities': similarity})


def reset_fingerprint_data(feature_size):
    if os.path.exists("fingerprint_features.npy"):
        os.remove("fingerprint_features.npy")
    if os.path.exists("fingerprint_labels.npy"):
        os.remove("fingerprint_labels.npy")
    np.save('fingerprint_features.npy', np.empty((0, feature_size)))
    np.save('fingerprint_labels.npy', np.array([]))

    if os.path.exists("fingerprint_features1.npy"):
        os.remove("fingerprint_features1.npy")
    if os.path.exists("fingerprint_labels1.npy"):
        os.remove("fingerprint_labels1.npy")
    np.save('fingerprint_features1.npy', np.empty((0, feature_size)))
    np.save('fingerprint_labels1.npy', np.array([]))


@app.route('/api/reset', methods=['POST'])
def reset_fingerprint():
    try:
        output_shape = model.output_shape
        feature_size = output_shape[-1]
        reset_fingerprint_data(feature_size)
        return jsonify({"message": "Fingerprint data has been reset successfully."}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['VEFIFY_FOLDER']):
        os.makedirs(app.config['VEFIFY_FOLDER'])
    app.run(debug=True)
