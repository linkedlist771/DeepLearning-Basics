import numpy as np
import matplotlib.pyplot as plt


def relu(X):
    '''
    relu 激活函数
    :param X: 输入
    :return:  输出
    '''
    a = np.copy(X)
    a[np.where(a<=0)] = 0
    return a



class MLP():
    def __init__(self,num_input, num_hidden, num_output):
        '''
        多层感知机结构
        :param : num_input(输入层个数) ,num_hidden(隐藏层个数),num_output(输出层个数)
        :return:  输出
        这里定义MLP为三层，分别为: 输入层 ,隐藏层，输出层
        其中，输入层，大小为1*m ,
             输出层为：m*1
        '''
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.W1 = np.random.random(size=(num_input, num_hidden))  # 第一层的权重
        self.b1 = np.zeros(shape=(1, num_hidden)) # 第一层偏置
        self.W2 = np.random.random(size=(num_hidden, num_output))  # 第二层的权重
        self.b2 = np.zeros(shape=(1, num_output)) # 第二层偏置
        self.activation_function = relu # 定义激活函数

    def forward(self, X):
        X = np.reshape(X,(1, self.num_input))
        H = self.activation_function(np.dot(X, self.W1)+self.b1)
        return np.dot(H, self.W2)+self.b2 # 返回这个值


if __name__ == '__main__':
    num_input = 10
    num_hidden = 20
    num_output = 1
    X = np.ones(shape=(1, num_input))
    model = MLP(num_input, num_hidden, num_output)
    print(f"开始的X:\n{X}")
    print(f"经过MLP前向传播的X:\n{model.forward(X)}")





