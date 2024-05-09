import numpy as np

# 加载图像和标签数据
def load_data(images_path, labels_path, num):
    images = []
    labels = []

    # 读取图像和标签数据
    with open(images_path, 'rb') as f_images:
        f_images.read(16)  # 跳过文件头
        for _ in range(num):
            image = np.frombuffer(f_images.read(28 * 28), dtype=np.uint8)
            image = image.reshape((28, 28))
            images.append(image)
        images_array = np.array(images)
        images_array = images_array.reshape(num, 28, 28)

    with open(labels_path, 'rb') as f_labels:
        f_labels.read(8)  # 跳过文件头
        for _ in range(num):
            label = np.frombuffer(f_labels.read(1), dtype=np.uint8)
            labels.append(label)
        labels_array = np.array(labels)
        labels_array = labels_array.reshape(num, 1)

    return images_array, labels_array
