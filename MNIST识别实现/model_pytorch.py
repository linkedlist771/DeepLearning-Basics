import torch.nn as nn
import torch


class MyNeuralNetWork(nn.Module):
    def __init__(self):
        super(MyNeuralNetWork, self).__init__()
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),
            nn.Softmax()

        )
        """ 
            nn.Conv2d(1, 28, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(28, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2,),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.Linear(64, 10)
        )"""

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':
    NN = MyNeuralNetWork()
    input = torch.ones((16, 1, 28, 28))
    output = NN(input)
    print(output.shape)
    # 这里是64个图片，输出的10分别对应的其的类别
