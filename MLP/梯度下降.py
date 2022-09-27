import numpy as np
from matplotlib import pyplot as plt

np.random.seed(100)  # 设置随机数种子
x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1)
plt.scatter(x, y)
plt.show()
# 初始化随机参数
w1 = np.random.rand(1, 1)
b1 = np.random.rand(1, 1)
# 训练模型
lr = 0.001


def loss_function(y, y_predict):  # real signature unknown; restored from __doc__
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
for i in range(800):
    # 前向传播
    y_pred = np.power(x, 2) * w1 + b1
    # loss
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    # 计算梯度
    #grad_w = np.sum((y_pred - y) * np.power(x, 2))
    #grad_b = np.sum((y_pred - y))
    # 使用梯度下降，min loss
    delta_theta = np.ones_like(w1) * 0.001  # 获得形状如X的值
    step_before = loss_function(y,(w1+delta_theta)* x**2+b1 )
    step_after = loss_function(y,(w1-delta_theta)* x**2+b1 )
    grad_w = (step_before - step_after) / (2 * delta_theta)  # 这里存在问题是不是应该取

    delta_theta = np.ones_like(b1) * 0.001  # 获得形状如X的值
    step_before = loss_function(y,w1* x**2+(b1+delta_theta) )
    step_after = loss_function(y,w1*x**2+(b1-delta_theta) )
    grad_b = (step_before - step_after) / (2 * delta_theta)  # 这里存在问题是不是应该取'''
    w1 -= lr * grad_w
    b1 -= lr * grad_b
# 可视化结果
plt.plot(x, y_pred, 'r-', label='predict')
plt.scatter(x, y, color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()
print(w1, b1)