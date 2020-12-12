# -*- coding: utf-8 -*-
# @Time    : 11/8/2020 3:46 PM
# @Author  : MLee
# @File    : __init__.py.py
import os

model_path = "mnist/model/mnist-cnn.pt"
if not os.path.isdir(os.path.dirname(model_path)):
    os.mkdir(os.path.dirname(model_path))
