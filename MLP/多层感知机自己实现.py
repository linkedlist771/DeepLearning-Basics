import numpy as np
import matplotlib.pyplot as plt
import pickle
import traceback
import  activation_function as f
from MLP_network import  *
def sigmoid(x):
    '''
    sigmoid 激活函数
    :param x: 输入
    :return:  输出
    '''

    return 1/(1+np.exp(-x))


def tanh(x):
    '''
    tanh 激活函数
    :param x: 输入
    :return:  输出
    '''

    return np.tanh(x)



def relu(x):
    '''
    relu 激活函数
    :param x: 输入
    :return:  输出
    '''
    a = np.copy(x)
    a[np.where(a <= 0)] = 0
    return a


def leaky_relu(x, r=0.01):
    '''
    LeakyRelu激活函数
    :param x: 输入
    :return:  输出
    '''
    a = np.copy(x)
    a[np.where(a <= 0)] *= r
    return a


def e_relu(x,  r=0.01):
    '''
    LeakyRelu激活函数
    :param x: 输入
    :return:  输出
    '''
    a = np.copy(x)
    a[np.where(a <= 0)] = r*(np.exp(a[np.where(a <= 0)])-1)
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
        self.activation_function = sigmoid  # 定义激活函数
        self.params = dict()
        self.params['W1'] = np.random.random(size=(num_input, num_hidden))  # 第一层的权重
        self.params['b1'] = np.zeros(shape=(1, num_hidden))  # 第一层偏置
        self.params['W2'] = np.random.random(size=(num_hidden, num_output))  # 第二层的权重
        self.params['b2'] = np.zeros(shape=(1, num_output))  # 第二层偏置
        self.theta = np.array([self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']], dtype=object)
    def forward(self, X):
        X = np.reshape(X,(-1, self.num_input))
        H = self.activation_function(np.dot(X, self.params['W1'])+self.params['b1'])
        return  np.dot(H, self.params['W2'])+ self.params['b2'] # 返回这个值

    def predict(self, X, theta):
        '''

        Parameters
        ----------
        X:输入值
        Returns
        -------
        输入推理值
        '''

        H = self.activation_function(np.dot(X, theta[0])+theta[1])#
        return  np.dot(H, theta[2])+theta[3]

    def activate(self, x, w, b, func):
        '''

        Parameters
        ----------
        x:输入x
        w:输入权重
        b:输入偏置
        func:所使用的激活函数的类

        Returns
        -------
        输出被激活的输出
        '''

        return func(np.dot(x, w)+b).forward()

    def loss_function(self, y, y_predict):
        '''

        Parameters
        ----------
        y:监督值
        y_predict:预测/推理值
        Returns
        -------
        返回两者的误差,这里采用的MSE误差
        '''
        return np.sum(0.5 * (y_predict - y) ** 2)

    def backward(self, X, Y, lr=0.0001):
        #前向赋值
        batch_size = len(X)
        nabla_w1 = []
        nabla_b1 = []
        nabla_w2 = []
        nabla_b2 = []
        for x,y in zip(X,Y):
            a2 = self.activate(x, self.params['W1'], self.params['b1'], f.Relu)
            a3 = self.activate(a2, self.params['W2'], self.params['b2'], f.Identity)
            #反向传播
            delta3 = a3*(1-a3)*(a3-y)
            delta2 = a2*(1-a2)*np.dot(delta3, self.params['W2'].T)
            if len(nabla_b1)==0:
                nabla_w1=np.dot(x.T, delta2)
                nabla_w2=np.dot(a2.T, delta3)
                nabla_b1=delta2
                nabla_b2=delta3
            else:
                nabla_w1+=np.dot(x.T, delta2)
                nabla_w2+=np.dot(a2.T, delta3)
                nabla_b1+=delta2
                nabla_b2+=delta3
        nabla_w1 /= batch_size
        nabla_w2 /= batch_size
        nabla_b1 /= batch_size
        nabla_b2 /= batch_size
            #梯度下降
        self.params['W1'] -= lr*nabla_w1
        self.params['W2'] -= lr*nabla_w2
        self.params['b1'] -= lr*nabla_b1
        self.params['b2'] -= lr*nabla_b2
        self.theta = np.array([self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']], dtype=object)
    def train(self, epoch):
        pass

    def sgd(self, X, y, delta = 1e-6 , lr=0.0001):
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

        theta = self.theta
        for index in range(len(theta)):
            para = theta[index]
            grad_t = np.zeros(shape=para.shape)  # 获得存储这个梯度的列表
            for i in range(len(grad_t.flatten())):
                delta_theta = np.array([np.zeros(shape = self.params['W1'].shape),
                                        np.zeros(shape = self.params['b1'].shape),
                                        np.zeros(shape = self.params['W2'].shape),
                                        np.zeros(shape = self.params['b2'].shape)], dtype=object)  # 获得形状如X的值
                shape = delta_theta[index].shape
                delta_theta_index = delta_theta[index].ravel()
                delta_theta_index[i] = delta
                delta_theta[index] = delta_theta_index.reshape(shape)
                step_before = self.loss_function(y, self.predict(X, theta+delta_theta))
                step_after = self.loss_function(y, self.predict(X, theta-delta_theta))  # 获得差分的分子
                grad = (step_before-step_after)/(2*delta)  # 这里存在问题是不是应该取
                shape = self.theta[index].shape
                self_theta = self.theta[index].ravel()
                self_theta[i] -= grad*lr
                self.theta[index] = self_theta.reshape(shape)
        self.params['W1'] = self.theta[0]
        self.params['b1'] = self.theta[1]
        self.params['W2'] = self.theta[2]
        self.params['b2'] = self.theta[3]

    def print_weight(self):
        print("W1")
        print(self.W1)
        print("b1")
        print(self.b1)
        print("W2")
        print(self.W2)
        print("b2")
        print(self.b2)

    # 保存权重文件    
    def save(self, path):
        with open(path, 'wb') as f :
            pickle.dump(self.params, f)

    # 导入权重文件
    def load(self, path):
        with open(path, 'rb') as f :
            self.params = pickle.load(f)
            self.theta = np.array([self.params['W1'], self.params['b1'], self.params['W2'],
                                   self.params['b2']], dtype=object)


def train_with_numeric_sgd():
    num_input = 1
    num_hidden = 30  # 设置隐藏层个数
    num_output = 1
    save_path = "weight.pickle"
    # 尝试读取保存的权重
    X = np.linspace(0, 2 * np.pi, 100).reshape((-1, num_input))
    y = np.sin(X)
    model = MLP(num_input, num_hidden, num_output)
    train_net = True
    load_weight = False
    if load_weight:
        try:
            model.load(save_path)  # 尝试读取权重
        except (Exception, BaseException) as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)
    epoch = 1000000  # 设置100次训练

    if train_net:
        for i in range(1, epoch + 1):
            model.sgd(X, y, lr=1e-4)
            if i % 100 == 0:
                print(f"第{i}/{epoch}次训练")
                print(f"训练误差{model.loss_function(y, model.forward(X))}")
                model.save(save_path)

    plt.figure()
    plt.scatter(X, y, color="black")
    plt.plot(X, model.forward(X), color="red")
    plt.title("MLP fits sin by hands")
    plt.legend(["original", "predicted"])
    plt.savefig("手动实现_MLP.png", dpi=300)
    plt.show()

def train_with_backward_propagation():
    # Loading the MNIST data
    X = np.linspace(0, 2 * np.pi, 100).reshape((-1,1,1))
    Y = np.sin(X)
    save_path = "weight_bp.pickle"
    training_data = list(zip(X, Y))
    net = Network([1, 30, 1])
    train_net = True
    load_weight = True
    #net.load(save_path)
    if load_weight:
        try:
            net.load(save_path)  # 尝试读取权重
        except (Exception, BaseException) as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)

    if train_net:
        net.SGD(training_data, epochs=100000, mini_batch_size=100000, eta=1e-4, test_data=training_data, save_path=save_path)
    plt.figure()
    plt.scatter(np.array(X).flatten(), np.array(Y).flatten() , color="black")
    predict_y = [net.feedforward(x) for x in  X]
    #print(predict_y.shape)
    plt.plot(np.array(X).flatten(), np.array(predict_y).flatten() , color="red")
    plt.title("MLP fits sin by hands")
    plt.legend(["original", "predicted"])
    plt.savefig("手动实现_MLP.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    #train_with_numeric_sgd()
    train_with_backward_propagation()