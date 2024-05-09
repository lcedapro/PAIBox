import numpy as np
from ubyte2npy import load_data

if __name__ == "__main__":
    # 加载数据
    images_array, labels_array = load_data('train-images.idx3-ubyte','train-labels.idx1-ubyte',60000)

    # 检查结果的形状
    print("图像数据形状：", images_array.shape)
    print("标签数据形状：", labels_array.shape)

    np.save("train-images.npy", images_array)
    np.save("train-labels.npy", labels_array)

    # 加载数据
    images_array, labels_array = load_data('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',10000)

    # 检查结果的形状
    print("图像数据形状：", images_array.shape)
    print("标签数据形状：", labels_array.shape)

    np.save("t10k-images.npy", images_array)
    np.save("t10k-labels.npy", labels_array)