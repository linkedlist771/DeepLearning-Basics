import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(1, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1))

    def forward(self, x):
        x = self.module(x)
        return x



plt.figure()


# 这里实现一个线性回归
#  设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把数据移动到cuda上
net = MLP()
net = net.to(device)
x = torch.arange(0, 20*np.pi, 0.01).reshape(-1, 1)
x = x.to(device)
y = torch.sin(x)#+20+torch.rand(size=(28, 1))*20
y = y.to(device)
epoch = 400000  # 十次训练次数
lr = 0.01  # 学习率
optimizer = torch.optim.Adam((net.parameters()), lr=lr )#SGD(net.parameters(),lr=lr)  #设置优化器
loss_function = nn.MSELoss()
train_net = False
load_weight = True

if load_weight:
    net.load_state_dict(torch.load('线性回归.pth'), strict=True)

if train_net:
  for i in range(1,epoch+1):
  
      output = net(x)
      loss = loss_function(output,y)
      optimizer.zero_grad()  #梯度清0
      loss.backward()
      optimizer.step()
      if i%100==0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9#注意这里
        print(f"第{i}次训练")
        print(f"训练误差{loss.item()}")
  
  torch.save(net.state_dict(),"线性回归.pth")


