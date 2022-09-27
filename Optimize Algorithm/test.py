import numpy as np
import matplotlib.pyplot as plt
class Optimizer:
    def __init__(self,
                 epsilon=1e-10,  # 误差
                 iters=100000,  # 最大迭代次数
                 lamb=0.01,  # 学习率
                 r=0.0,  # 累积梯度
                 theta=1e-7):  # 常数
        self.epsilon = epsilon
        self.iters = iters
        self.lamb = lamb
        self.r = r
        self.theta = theta

    def adagrad(self, x_0=0.5, y_0=0.5):
        f1, f2 = self.fn(x_0, y_0), 0
        w = np.array([x_0, y_0])  # 每次迭代后的函数值，用于绘制梯度曲线
        k = 0  # 当前迭代次数

        while True:
            if abs(f1 - f2) <= self.epsilon or k > self.iters:
                break
            print(x_0,y_0)
            print(self.r)
            f1 = self.fn(x_0, y_0)

            g = np.array([self.dx(x_0, y_0), self.dy(x_0, y_0)])
            self.r += np.dot(g, g)
            x_0, y_0 = np.array([x_0, y_0]) - self.lamb / (self.theta + np.sqrt(self.r)) * np.array(
                [self.dx(x_0, y_0), self.dy(x_0, y_0)])
            f2 = self.fn(x_0, y_0)

            w = np.vstack((w, (x_0, y_0)))
            k += 1

        self.print_info(k, x_0, y_0, f2)
        self.draw_process(w)

    def print_info(self, k, x_0, y_0, f2):
        print('迭代次数：{}'.format(k))
        print('极值点：【x_0】：{} 【y_0】：{}'.format(x_0, y_0))
        print('函数的极值：{}'.format(f2))

    def draw_process(self, w):
        X = np.arange(0, 1.5, 0.01)
        Y = np.arange(-1, 1, 0.01)
        [x, y] = np.meshgrid(X, Y)
        f = x ** 3 - y ** 3 + 3 * x ** 2 + 3 * y ** 2 - 9 * x
        plt.contour(x, y, f, 20)
        plt.plot(w[:, 0], w[:, 1], 'g*', w[:, 0], w[:, 1])
        plt.show()

    def fn(self, x, y):
        return x**2/20+y**2

    def dx(self, x, y):
        return x/10

    def dy(self, x, y):
        return 2*y
o = Optimizer()
o.adagrad()