#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 10:46
# @Author  : xiezheng
# @Site    : 
# @File    : pytorchToonnx.py


import torch
import torchvision
import torch.onnx


if __name__ == '__main__':

    # An instance of your model
    model = torchvision.models.resnet18()

    # An example input you would normally provide to your model's forward() method
    x = torch.rand(1, 3, 224, 224)

    # Export the model
    torch_out = torch.onnx._export(model, x, "resnet18.onnx", export_params=True)

