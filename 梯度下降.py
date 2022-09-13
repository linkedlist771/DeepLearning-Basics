import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


# 构造数据
def get_data(sample_num=10000):
    """
    拟合函数为
    y = 5*x1 + 7*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([5, 7]).T)
    return x, y


# 梯度下降法
def GD(samples, y, step_size=0.01, max_iter_count=100):
    """
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值
    m, var = samples.shape
    theta = np.zeros(2)
    y = y.flatten()
    # 进入循环内
    print(samples)
    loss = 1
    iter_count = 0
    iter_list = []
    loss_list = []
    theta1 = []
    theta2 = []
    # 当损失精度大于0.01且迭代此时小于最大迭代次数时，进行
    while loss > 0.01 and iter_count < max_iter_count:
        loss = 0
        # 梯度计算
        theta1.append(theta[0])
        theta2.append(theta[1])
        for i in range(m):
            h = np.dot(theta, samples[i].T)
            # 更新theta的值,需要的参量有：步长，梯度
            for j in range(len(theta)):
                theta[j] = theta[j] - step_size * (1 / m) * (h - y[i]) * samples[i, j]
        # 计算总体的损失精度，等于各个样本损失精度之和
        for i in range(m):
            h = np.dot(theta.T, samples[i])
            # 每组样本点损失的精度
            every_loss = (1 / (var * m)) * np.power((h - y[i]), 2)
            loss = loss + every_loss

        print("iter_count: ", iter_count, "the loss:", loss)

        iter_list.append(iter_count)
        loss_list.append(loss)

        iter_count += 1
    plt.plot(iter_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return np.array(theta1), np.array(theta2), np.array(theta), np.array(loss_list)


def painter3D(theta1, theta2, loss):
    style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    x, y, z = theta1, theta2, loss
    ax1.plot_wireframe(x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1), rstride=5, cstride=5)
    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta2")
    ax1.set_zlabel("loss")
    plt.show()


def predict(x, theta):
    y = np.dot(theta, x.T)
    return y


if __name__ == '__main__':
    samples, y = get_data()
    theta1, theta2, theta, loss_list = GD(samples, y)
    print(theta)  # 会很接近[5, 7]
    painter3D(theta1, theta2, loss_list)
    predict_y = predict(theta, [7, 8])
    print(predict_y)