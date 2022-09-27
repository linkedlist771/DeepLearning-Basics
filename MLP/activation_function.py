import numpy as np


class Sigmoid:
    def __init__(self, x):
        self.x = x

    def forward(self):
        '''

        Returns
        -------
        返回正向计算得到的值
        '''

        return 1/(1+np.exp(-self.x))

    def backward(self):
        '''

        Returns
        -------
        返回sigmoid激活函数的求导后的值
        '''

        return self.forward()-np.square(self.forward())


class TanH:
    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.tanh(self.x)

    def backward(self):
        return np.square(np.sinh(self.x))


class Relu:
    def __init__(self, x):
        self.x = x

    def forward(self):
        a = np.copy(self.x)
        a[np.where(a <= 0)] = 0
        return a

    def backward(self):
        a = np.zeros_like(self.x)
        a[np.where(a > 0)] = 1
        return a


class LeakyRelu:
    def __init__(self, x, r=0.01):
        self.x = x
        self.r = r

    def forward(self):
        a = np.copy(self.x)
        a[np.where(a <= 0)] *= self.r
        return a

    def backward(self):
        a = np.ones_like(self.x)
        a[np.where(a <= 0)] = self.r
        return a


class ERelu:
    def __init__(self, x, r=0.01):
        self.x = x
        self.r = r

    def forward(self):
        a = np.copy(self.x)
        a[np.where(a <= 0)] = self.r * (np.exp(a[np.where(a <= 0)]) - 1)
        return a

    def backward(self):
        a = np.ones_like(self.x)
        a[np.where(a <= 0)] = self.r * np.exp(a[np.where(a <= 0)])
        return a

class Identity:
    def __init__(self, x):
        self.x = x

    def forward(self):
        '''

        Returns
        -------
        返回正向计算得到的值
        '''

        return self.x

    def backward(self):
        '''

        Returns
        -------
        返回sigmoid激活函数的求导后的值
        '''

        return np.ones_like(self.x)