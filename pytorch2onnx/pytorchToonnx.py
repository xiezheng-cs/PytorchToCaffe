#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 10:46
# @Author  : xiezheng
# @Site    : 
# @File    : pytorchToonnx.py


import torch
import torchvision
from torch import onnx
# import onnx

from pruned_mobilenetv2 import PrunedMobileNetV2
from pruned_mobilenetv1 import PrunedMobileNetV1


if __name__ == '__main__':

    # mobilenetv1_baseline
    # output_onnx = './onnx_model/mobilenetv1.onnx'
    # model = PrunedMobileNetV1(pruning_rate=0.0)
    # checkpoint_path = '/home/liujing/NFS/TPAMI_channel_pruning/baseline/imagenet_mobilenetv1/log_aux_mobilenetv1_imagenet_imagenet_bs256_e90_lr0.180_step[30, 60, 90]_baseline_official_fix_2019052201/check_point/checkpoint_089.pth'
    # mobilenetv1_p0.3
    output_onnx = './onnx_model/mobilenetv1_p0.3.onnx'
    model = PrunedMobileNetV1(pruning_rate=0.3)
    checkpoint_path = ''

    # mobilenetv2_baseline
    # output_onnx = './onnx_model/mobilenetv2.onnx'
    # model = PrunedMobileNetV2(pruning_rate=0.0)
    # checkpoint_path = '/home/liujing/NFS/TPAMI_channel_pruning/baseline/imagenet_mobilenetv2/log_aux_mobilenetv2_imagenet_imagenet_bs256_e150_lr0.050_step[30, 60, 90]_baseline_original_2019060901/check_point/checkpoint_149.pth'
    # mobilenetv2_p0.3
    # output_onnx = './onnx_model/mobilenetv2_p0.3.onnx'
    # model = PrunedMobileNetV2(pruning_rate=0.3)
    # checkpoint_path = '/home/liujing/NFS/TPAMI_channel_pruning/finetune/mobilenetv2_imagenet/explore_pruning_rate/log_aux_mobilenetv2_imagenet_imagenet_bs96_e250_lr0.180_step[]_p0.3_nowarmstart_cosine_2019081901/check_point/checkpoint_249.pth'

    checkpoint = torch.load(checkpoint_path)
    # print(model)
    model.load_state_dict(checkpoint['model'])
    print('model load success!!!')

    # An example input you would normally provide to your model's forward() method
    x = torch.randn(1, 3, 224, 224)
    input_names = ["data"]

    # Export the model
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    torch_out = torch.onnx._export(model, x, output_onnx, export_params=True, input_names=input_names)
    # torch_out = torch.onnx._export(model, x, output_onnx, export_params=True)
    print('pytorchToonnx finished!!!')

