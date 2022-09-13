import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    '''
    sigmoid 激活函数
    :param X: 输入
    :return:  输出
    '''

    return 1/(1+np.exp(-X))


def tanh(X):
    '''
    tanh 激活函数
    :param X: 输入
    :return:  输出
    '''

    return np.tanh(X)



def relu(X):
    '''
    relu 激活函数
    :param X: 输入
    :return:  输出
    '''
    a = np.copy(X)
    a[np.where(a<=0)] = 0
    return a


def leaky_relu(X, r=0.01):
    '''
    LeakyRelu激活函数
    :param X: 输入
    :return:  输出
    '''
    a = np.copy(X)
    a[np.where(a<=0)] *= r
    return a


def e_relu(X, r=0.01):
    '''
    LeakyRelu激活函数
    :param X: 输入
    :return:  输出
    '''
    a = np.copy(X)
    a[np.where(a<=0)] = r*(np.exp(a[np.where(a<=0)])-1)
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
        X = np.reshape(X,(-1, self.num_input))
        H = self.activation_function(np.dot(X, self.W1)+self.b1)
        return np.dot(H, self.W2)+self.b2 # 返回这个值


    def predict(self, X, theta):
        '''

        Parameters
        ----------
        X:输入值
        Returns
        -------
        输入推理值
        '''

        H = self.activation_function(np.dot(X, theta[0])+theta[1])#np.array([self.W1, self.b1, self.W2, self.b2])
        return np.dot(H, theta[2])+theta[3]


    def loss_function(self, y, y_predict):# real signature unknown; restored from __doc__
        '''

        Parameters
        ----------
        y:监督值
        y_predict:预测/推理值

        Returns
        -------
        返回两者的误差,这里采用的MSE误差
        '''
        return np.sum(np.square(y-y_predict))/len(y)


    def get_gradient(self,  X, y, delta=0.01):
        '''

        Parameters
        ----------
        X:输入值
        y:监督值
        y_predict:预测/推理值
        delta:步长

        Returns
        -------
        返回梯度
        '''

        theta = np.array([self.W1, self.b1, self.W2, self.b2])
        delta_theta = np.ones_like(theta)*delta  # 获得形状如X的值
        step_before = self.loss_function(y, self.predict(X, theta+delta_theta))
        step_after = self.loss_function(y, self.predict(X, theta-delta_theta))  # 获得差分的分子
        return (step_before-step_after)/(2*delta_theta)  # 这里存在问题是不是应该取



    def sgd(self, X, y, lr=0.0001):
        '''
        用于做SGD优化的权重以及偏置参数即为__init__
        里面的参数
        Parameters
        ----------
        X: 输入参数
        y: 监督值
        lr: 学习率
        Returns
        -------
        无返回值，仅仅优化权重以及偏置参数
        '''

        gradient = self.get_gradient(X, y)
        self.W1 -= gradient[0]*lr
        self.b1 -= gradient[1]*lr
        self.W2 -= gradient[2]*lr
        self.b2 -= gradient[3]*lr
    def train(self):
        pass






if __name__ == '__main__':
    num_input = 1
    num_hidden = 128
    num_output = 1
    X = np.arange(0, 2*np.pi, 0.1).reshape((-1, num_input))
    y = np.sin(X)*20
    model = MLP(num_input, num_hidden, num_output)
    epoch = 10000 # 设置100次训练
    for i in range(1,epoch+1):
        model.sgd(X,y)
        if i%1000==0:
            print(f"第{i}次训练")
            print(f"训练误差{model.loss_function(y,model.forward(X))}")
            plt.figure()
            plt.scatter(X, y, color="black")
            plt.plot(X, model.forward(X), color="red")
            plt.title("MLP fits sin by hands")
            plt.legend(["original", "predicted"])
    #print(f"开始的X:\n{X}")
    #print(f"经过MLP前向传播的X:\n{model.forward(X)}")
    plt.scatter(X, y, color="black")
    plt.plot(X, model.forward(X), color="red")
    plt.title("MLP fits sin by hands")
    plt.legend(["original","predicted"])
    plt.savefig("手动实现_MLP.png",dpi=300)
    plt.show()




