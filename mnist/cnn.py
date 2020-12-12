# -*- coding: utf-8 -*-
# @Time    : 11/8/2020 3:46 PM
# @Author  : MLee
# @File    : cnn.py
import functools
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# SGD 优化器，学习率: 0.001
sgd_optimizer = functools.partial(torch.optim.SGD, lr=0.001)


class CNNModel(nn.Module):
    """
    CNN model for mnist
    """

    def __init__(self, num_classes=10, optimizer=sgd_optimizer, device=None):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.device = 'cpu' if not device else device

        self.create_model(1)

        self.optimizer = optimizer(self.parameters())
        self.loss = F.nll_loss


    def create_model(self, num_channels=1):
        """
        Model function for CNN
        实现参考: https://github.com/pytorch/examples/blob/master/mnist/main.py
        :param num_channels: 图像 channels，MNIST 数据集为 1 通道的
        :return:
        """
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def set_params(self, model_params):
        self.load_state_dict(model_params)

    def get_params(self):
        return self.state_dict()

    def train_model(self, data, num_epochs=1, batch_size=32, verbose=False):
        """
        模型训练
        :param data: 训练数据集
        :param num_epochs:
        :param batch_size:
        :param verbose: 是否打印训练过程
        :return: 模型参数
        """
        self.train()  # set train mode

        model = self.to(self.device)
        for epoch in range(num_epochs):
            for index, (X, y) in enumerate(DataLoader(data, batch_size=batch_size, shuffle=True)):
                source, target = X.to(self.device), y.to(self.device)
                self.zero_grad()
                output = model(source)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                if verbose and index % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, index * batch_size, len(data), 100. * (index * batch_size) / len(data), loss.item()))

        return self.get_params()

    def test(self, data):
        """
        用测试数据 test 模型
        :param data:
        :return: (correct, test_loss)
            correct: 预测正确样本数
            test_loss: 总测试 loss 值
        """
        self.eval()
        model = self.to(self.device)

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in DataLoader(data, batch_size=1000):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data)

        return correct, test_loss

    def save(self, save_path, parameters=None):
        """
        保存当前模型参数到指定文件
        :param save_path:
        :return:
        """
        assert os.path.isdir(os.path.dirname(save_path)), "The target path doesn't exist: {}".format(save_path)
        parameters = self.get_params() if not parameters else parameters
        torch.save(parameters, save_path)

    def load(self, model_path):
        """
        从文件导入模型参数
        :param model_path:
        :return:
        """
        assert os.path.isfile(model_path), "Model path doesn't exist: {}".format(model_path)
        self.set_params(torch.load(model_path))

    @property
    def model(self):
        """
        返回模型，以用于预测推理
        :return:
        """
        return self.to(self.device)

    def predict(self, x: torch.Tensor):
        """
        对输入数据 x，进行预测
        :param x:
        :return: mnist 对应的 class index
        """
        output = self.model(x.to(self.device))
        return output.argmax(dim=1, keepdim=False).to("cpu").item()
