from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Đường dẫn tới thư mục chứa hình ảnh vân tay
data_dir = r'Data\SOCOFing\Real'

# Hàm để tải và xử lý hình ảnh


def load_images(data_dir):
    images = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.BMP'):
            # print(file_name)
            # print(file_name.split('__')[0])
            # print(os.path.join(data_dir, file_name))
            img = cv2.imread(os.path.join(
                data_dir, file_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (96, 96))
            images.append(img)
            label = int(file_name.split('__')[0])
            labels.append(label)
    return np.array(images), np.array(labels)


# def test(data_dir):
#     file = open('file_name.txt', 'w')
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith('.BMP'):
#             file.write(file_name + "\n")
#     file.close()


# test(data_dir)

def display_sample_images(images, labels, number_samples=5):
    plt.figure(figsize=(10, 10))
    for i in range(number_samples):
        index = np.random.randint(0, len(images))
        img = images[index]
        label = labels[index]
        plt.subplot(1, number_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"label: {label}")
        plt.axis()
    plt.show()


images, labels = load_images(data_dir)
display_sample_images(images, labels)

# # Chuẩn hóa hình ảnh
# images = images / 255.0
# images = images.reshape(-1, 96, 96, 1)

# # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(
#     images, labels, test_size=0.2, random_state=42)


# num_classes = len(np.unique(labels))

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())

# history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)


# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
