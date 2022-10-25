import matplotlib.pyplot as plt
import numpy as np
import torch
import struct
import cv2
from model_pytorch import *


def load_mnist_data(kind):
    '''
    加载数据集
    :param kind: 加载训练数据还是测试数据
    :return: 打平之后的数据和one hot编码的标签
    '''
    labels_path = 'data/%s-labels-idx1-ubyte' % kind
    images_path = 'data/%s-images-idx3-ubyte' % kind
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images / 255., np.eye(10)[labels]


model = MyNeuralNetWork()
model.load_state_dict(torch.load('Mnist Epoch Best.pth'))
test_images, test_labels = load_mnist_data('t10k')
"""
for img, label in zip(test_images, test_labels):
    img = torch.Tensor(img)
    label = torch.Tensor(label.reshape(1, -1))
    img = img.reshape(28, 28)
    plt.imshow(img, cmap="gray")
    img = img.reshape(-1, 784)
    output = model(img).argmax(1)
    plt.title(f"Real: {label.argmax(1).data[0]}, \n predict: {output.data[0]}")
    plt.show()

"""
img = plt.imread("4.png")
img = 1-np.dot(img, [0.299, 0.587, 0.114])
plt.imshow(img, cmap="gray")
plt.show()
img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_LANCZOS4)
plt.imshow(img, cmap="gray")
img = img.reshape(-1, 784)
img = torch.Tensor(img)
output = model(img)
print(output)
print(output.argmax(1))
plt.title(f"Real: {4}, \n predict: {output.argmax(1).data[0]}")
plt.show()
