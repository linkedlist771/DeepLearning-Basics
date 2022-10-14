"""
这个是使用pytorch版本来实现手写数据集分类
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import struct
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

def show_image():
    plt.imshow(train_images[0].reshape(28, 28))
    plt.show()


test_images, test_labels = load_mnist_data('t10k')
test_images = np.reshape(test_images, newshape=(-1, 1, 28, 28))
test_images = torch.Tensor(test_images)
test_labels = torch.Tensor(test_labels)
# 构建神经网络
model = MyNeuralNetWork()
# 创建损失函数
loss_function = nn.CrossEntropyLoss()
# 每次取出的图片数量
batch_size = 128
n_test = len(test_images)
# 每训练一个epoch查看测试误差
total_loss = 0
# 进行测试，不需要梯度
model.load_state_dict(torch.load('Mnist Epoch Best.pth'))
with torch.no_grad():
        accuracy = 0
        for batch_index in range(0, n_test, batch_size):
            lower_range = batch_index
            upper_range = batch_index + batch_size
            if upper_range > n_test:
                upper_range = n_test
            images = test_images[lower_range: upper_range, :]
            labels = test_labels[lower_range: upper_range]
            output = model(images)
            loss = loss_function(output, labels)
            total_loss += loss.item()
            accuracy += (output.argmax(1) == labels.argmax(1)).sum()
        print(f"在整体测试集上的Loss:{total_loss}")
        print(f"在整体测试集上的正确率:{accuracy/len(test_images)*100}%")