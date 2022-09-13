import torch
from torch import nn
import  matplotlib.pyplot as plt
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(1, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1))

    def forward(self, x):
        x = self.module(x)
        return x



plt.figure()


# 这里实现一个线性回归
net = MLP()
x = torch.reshape(torch.range(1, 28), (28, 1))
y = 20*x+20+torch.rand(size=(28, 1))*20
epoch = 10000  # 十次训练次数
lr = 0.0001  # 学习率
optimzer = torch.optim.Adam((net.parameters()),lr=lr)#SGD(net.parameters(),lr=lr)  #设置优化器
loss_function = nn.MSELoss()
for i in range(1,epoch+1):

    output = net(x)
    loss = loss_function(output,y)
    optimzer.zero_grad()  #梯度清0
    loss.backward()
    optimzer.step()
    if i%100==0:
      print(f"第{i}次训练")
      print(f"训练误差{loss.item()}")

torch.save(net.state_dict(),"线性回归.pth")


with torch.no_grad():
    plt.scatter(x, y, color="black")
    plt.plot(x, net(x), color="red")
    plt.legend(["original","predicted"])
    plt.savefig("多层感知机线性回归.png",dpi=300)
    plt.show()

