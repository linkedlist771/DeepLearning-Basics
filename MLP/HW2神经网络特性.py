import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt



def init_params(nn, distribution,delta = 1.0):
    nn = nn
    if distribution=="noraml":
        nn.weight.data.normal_(0.0, delta)
        nn.bias.data.fill_(0)
    elif distribution=="0":
        nn.weight.data.fill_(0)
        nn.bias.data.fill_(0)
    else:
        assert "There is no such neural network!"
    return nn


class MLP(nn.Module):
    def __init__(self, distribution, delta):
        super(MLP, self).__init__()
        self.activation_func = nn.Sigmoid()
        self.layer1_theta = init_params(nn.Linear(100, 100), distribution, delta)
        self.layer2_theta = init_params(nn.Linear(100, 100), distribution, delta)
        self.layer3_theta = init_params(nn.Linear(100, 100), distribution, delta)
        self.layer4_theta = init_params(nn.Linear(100, 100), distribution, delta)
        self.layer5_theta = init_params(nn.Linear(100, 100), distribution, delta)
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.a5 = None
        ''' 
        self.module = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(1, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1))'''

    def forward(self, x):
        self.a1 = self.activation_func(self.layer1_theta(x))
        self.a2 = self.activation_func(self.layer2_theta(self.a1))
        self.a3 = self.activation_func(self.layer3_theta(self.a2))
        self.a4 = self.activation_func(self.layer4_theta(self.a3))
        self.a5 = self.activation_func(self.layer5_theta(self.a4))
        return



def plot_hist(x, name,text):
    import seaborn as sns
    import os
    # 这里是对画布的设置。num 是 数字或者字符串，是figure的名字。
    # subplots是方便多个子图（只有一个图的时候写1,1就行）
    fig, axes = plt.subplots(1, 1, num=name, figsize=(10, 8))  # , sharex=True)
    # 使用画布背景。
    plt.style.use('seaborn-darkgrid')  # 'seaborn-bright'
    # 调色板，可以使用里面的颜色 color，还挺好看的
    palette = plt.get_cmap('tab20c')  # 'Pastel2') # 'Set1'
    plt.subplot(1, 1, 1)
    # 调用histplot作图
    ax1 = sns.histplot(x, kde=True, bins=100, shrink=1, color=palette.colors[0],
                       edgecolor=palette.colors[-1])  # "none")#, element="step")# element = "poly") # cumulative= True)
    # ax.invert_yaxis()
    # plt.gca().invert_yaxis()
    #36
    # newx = ax.lines[0].get_ydata()
    # newy = ax.lines[0].get_xdata()
    # # set new x- and y- data for the line
    # ax.lines[0].set_xdata(newx)
    # ax.lines[0].set_ydata(newy)
    # plt.subplot(2, 1, 2)
    # sns.distplot(df["stars"], kde = True, bins = 20)# element="step", fill=False)
    # 给直方图添加文字注释（就是在每一个bar的上方加文字）
    index = 0
    for p in ax1.patches:
        if p.get_height() > 0 and index%5==0:
            ax1.annotate(
            # 文字内容
            text = f"{p.get_height():1.0f}",
            # 文字的位置
            xy = (p.get_x() + p.get_width() / 2., p.get_height()),
            xycoords = 'data',
            ha = 'center',
            va = 'center',
            fontsize = 10,
            color = 'black',
            # 文字的偏移量
            xytext = (0, 7),
            textcoords = 'offset points',
            clip_on = True,  # <--- important
            )
            # 紧密排版

        index+=1
    text = text+f"<h2>取值:{name}</h2>\n<img src = '{name}.png'>\n <br> <br><br>"
    plt.title(name)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f"{name}.png")
    plt.show()
    return text

if __name__ == "__main__":
    text = ""
    activate_func = "Sigmoid"
    for theta in ["normal", "0"]:
        for delta in [1, 0.01]:
            x = torch.normal(mean=0, std=1, size=(1000, 100))
            net = MLP(theta, delta = delta)
            with torch.no_grad():
                print(1)
                net.forward(x)
                text = plot_hist(net.a1.flatten(), name=f"layer=a1, f={activate_func},θ={theta}, delta={delta}",text=text)#plt.hist(, bins=[0, 25, 50, 75, 100])
                text = plot_hist(net.a2.flatten(), name=f"layer=a2, f={activate_func},θ={theta}, delta={delta}",text=text)#plt.hist(, bins=[0, 25, 50, 75, 100])
                text = plot_hist(net.a3.flatten(), name=f"layer=a3, f={activate_func},θ={theta}, delta={delta}",text=text)#plt.hist(, bins=[0, 25, 50, 75, 100])
                text = plot_hist(net.a4.flatten(), name=f"layer=a4, f={activate_func},θ={theta}, delta={delta}",text=text)#plt.hist(, bins=[0, 25, 50, 75, 100])
                text = plot_hist(net.a5.flatten(), name=f"layer=a5, f={activate_func},θ={theta}, delta={delta}",text=text)#plt.hist(, bins=[0, 25, 50, 75, 100])
    with open("image_path.txt2","w") as f:
        f.write(text)
                #print()'''
                #print(net.layer1_theta.weight.data)