# -*- coding: utf-8 -*-
# @Time    : 11/8/2020 4:10 PM
# @Author  : MLee
# @File    : dataset.py

import os

from torchvision import transforms, datasets


class Dataset(object):

    def __init__(self, data_path='mnist/data'):
        super(Dataset, self).__init__()

        self.data_path = data_path
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_sets = datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        self.test_sets = datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)

    @property
    def classes(self):
        return self.test_sets.classes
