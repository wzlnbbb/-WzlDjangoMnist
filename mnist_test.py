# -*- coding: utf-8 -*-
# @Time    : 11/8/2020 4:10 PM
# @Author  : MLee
# @File    : mnist_test.py
import os

import torch
from PIL import Image
from torchvision import transforms

import mnist
from mnist.cnn import CNNModel
from mnist.dataset import Dataset

mnist_dataset = Dataset()
device = torch.device('cpu')
mnist_cnn = CNNModel(device=device)


def train():
    """
    模型训练
    :return:
    """
    mnist_cnn.train_model(mnist_dataset.train_sets, 100, verbose=True)
    mnist_cnn.save(mnist.model_path)
    correct, test_loss = mnist_cnn.test(mnist_dataset.test_sets)
    print("Test acc: {}% loss: {}".format(100. * correct / len(mnist_dataset.test_sets), test_loss))


def reload_model():
    """
    加载训练好的模型参数，并测试准确率
    :return:
    """
    mnist_cnn.load(mnist.model_path)
    correct, test_loss = mnist_cnn.test(mnist_dataset.test_sets)
    print("Test acc: {}% loss: {}".format(100. * correct / len(mnist_dataset.test_sets), test_loss))


def predict(image):
    """
    对输入图片，进行识别
    :param image:
    :return:
    """
    assert os.path.isfile(image), "Source image doesn't exist"
    mnist_cnn.load_state_dict(torch.load(mnist.model_path, map_location=device))

    # 读取图片文件，并将图片大小转换为 28*28，进行 Normalization
    pil_img = Image.open(image)
    pil_img = pil_img.convert('L')  # 转换为灰度图
    transform1 = transforms.Compose([
        transforms.Resize((28, 28)),  # 将图像分辨率转换为目标分辨率，只能对PIL图片进行裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # 图像正则化
    ])
    input_data = transform1(pil_img)
    input_data = input_data.unsqueeze(1)  # 加上 channel 信息

    pred = mnist_cnn.predict(input_data)
    # assert pred == 3, "Predict Failed"
    pred_class = mnist_dataset.classes[pred]
    print("Predict result: {}".format(pred_class))
    return pred_class


if __name__ == '__main__':
    predict("media/img/test_img.png")
