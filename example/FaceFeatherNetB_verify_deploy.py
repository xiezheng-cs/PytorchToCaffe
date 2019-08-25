# 2018.09.06 by Shining 
import sys
sys.path.insert(0,'/home/shining/Projects/github-projects/caffe-project/caffe/python')
import caffe
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.models import resnet
import time
from easydict import EasyDict

import cv2
# from FaceFeatherNetB import FaceFeatherNetB
from FaceFeatherNetB_v2 import FaceFeatherNetB_v2

# caffe load formate
def load_image_caffe(imgfile):
    print('imgfile={}'.format(imgfile))
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image


# pytorch load formate
def load_image_pytorch(imgfile):
    print('imgfile={}'.format(imgfile))
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)# 读取图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    v_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape((1, 1, 3))
    img = img.astype(np.float32) - v_mean
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = img.reshape([1,3,args.height, args.width])
    print("img = ",img)
    return img


def load_ones_numpy():
    img = np.ones([1,3,args.height, args.width])*10
    print("img = ", img)
    # 转化为numpy.ndarray并显示
    return img


# 全局变量，用于存储中间层的 feature
module_list = []
total_feat_out = []
total_feat_in = []

# 定义 forward hook function
def hook_fn_forward(module, input, output):
    # print(module) # 用于区分模块
    # print('input', input) # 首先打印出来
    # print('output', output)

    module_list.append(module)
    total_feat_out.append(output)  # 然后分别存入全局 list 中
    total_feat_in.append(input)


def forward_pytorch(weightfile, image, se):

    if se:
        net = FaceFeatherNetB_v2()
    else:
        net = FaceFeatherNetB_v2(se=False)

    checkpoint = torch.load(weightfile)
    net.load_state_dict(checkpoint['model'])
    print('pytorch load model success!!!')
    # assert False
    if args.cuda:
        net.cuda()
    # print(net)
    net.eval()
    image = torch.from_numpy(image).float()
    if args.cuda:
        image = image.cuda()
    else:
        image = image

    # net.features[0][0].register_forward_hook(hook_fn_forward)
    # net.features[0][0].register_forward_hook(hook_fn_forward)
    # net.features[1].conv[3].register_forward_hook(hook_fn_forward)

    t0 = time.time()
    blobs = net.forward(image)
    #print(blobs.data.numpy().flatten())
    t1 = time.time()
    # return t1-t0, blobs, net.parameters()

    return t1 - t0, blobs, net.named_parameters()


# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)

    # net.blobs['data'].reshape(1, 3, args.height, args.width)
    net.blobs['data'].data[...] = image

    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params


def save_model_state(file1, file2, pytorch_params, caffe_params):

    f1 = open(file1, 'w+')
    f2 = open(file2, 'w+')

    f1.write("pytorch_models\n")
    for name, param in pytorch_params:
        f1.write('name={}, param={}\n'.format(name, param))

    f2.write("caffe_params\n")
    # 遍历每一层
    for param_name in caffe_params.keys():
        # 权重参数
        weight = caffe_params[param_name][0].data
        # 偏置参数
        if len(caffe_params[param_name]) != 1:
            bias = caffe_params[param_name][1].data

        # 该层在prototxt文件中对应“top”的名称
        f2.write(param_name)
        f2.write('\n')

        # 写权重参数
        f2.write('\n' + param_name + '_weight:\n\n')
        # 权重参数是多维数组，为了方便输出，转为单列数组
        weight.shape = (-1, 1)

        for w in weight:
            f2.write('%ff, ' % w)

        # 写偏置参数
        f2.write('\n\n' + param_name + '_bias:\n\n')
        # 偏置参数是多维数组，为了方便输出，转为单列数组
        if len(caffe_params[param_name]) != 1:
            bias.shape = (-1, 1)
            for b in bias:
                f2.write('%ff, ' % b)
        f2.write('\n\n')

    f1.close()
    f2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    # 186
    # parser.add_argument('--se', default=False, type=bool)
    # parser.add_argument('--protofile', default='/home/xiezheng/lcr/xiezheng_tools/PytorchToCaffe/example/feathernetb_nose_prelu_checkpoint2.prototxt', type=str)
    # parser.add_argument('--weightfile', default='/home/xiezheng/lcr/xiezheng_tools/PytorchToCaffe/example/feathernetb_nose_prelu_checkpoint2.caffemodel', type=str)
    # parser.add_argument('--model', default="/home/xiezheng/lcr/model/nose_prelu_2_best.pth.tar", type=str)

    parser.add_argument('--se', default=True, type=bool)
    parser.add_argument('--protofile',
                        default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/FaceFeatherNetB_se_prelu_checkpoint4.prototxt',
                        type=str)
    parser.add_argument('--weightfile',
                        default='/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/FaceFeatherNetB_se_prelu_checkpoint4.caffemodel',
                        type=str)
    parser.add_argument('--model', default="/home/xiezheng/lcr/model/se_prelu_4_best.pth.tar", type=str)

    parser.add_argument('--imgfile', default='./example/2016.bmp', type=str)
    parser.add_argument('--height', default=112, type=int)
    parser.add_argument('--width', default=112, type=int)
    parser.add_argument('--meanB', default=127.5, type=float)
    parser.add_argument('--meanG', default=127.5, type=float)
    parser.add_argument('--meanR', default=127.5, type=float)
    parser.add_argument('--scale', default=0.0078125, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', default=False, help='enables cuda')

    args = parser.parse_args()
    print(args)

    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    image = load_image_pytorch(imgfile)
    # image = load_ones_numpy()

    time_pytorch, pytorch_blobs, pytorch_params = forward_pytorch(args.model, image, args.se)

    # assert False
    time_caffe, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, image)

    print(args)
    print('imge={}'.format(image))
    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')

    if args.se:
        blob_name = 'view_blob9'
    else:
        blob_name = "view_blob1"   # no_se

    if args.cuda:
        pytorch_data = pytorch_blobs.data.cpu().numpy().flatten()
    else:
        pytorch_data = pytorch_blobs.data.numpy().flatten()

    # caffe_input_1 = caffe_blobs['data'].data
    # caffe_output_1 = caffe_blobs['conv_blob1'].data
    # caffe_output_1 = caffe_blobs['conv_blob3'].data
    # print('module={}\npytorch input={}\ncaffe_input={}'.format(module_list, total_feat_in, caffe_input_1))
    # print('module={}\npytorch output={}\ncaffe_output={}'.format(module_list, total_feat_out, caffe_output_1))

    caffe_data = caffe_blobs[blob_name].data[0][...].flatten()
    diff = abs(pytorch_data - caffe_data).sum()
    print('pytorch out={}\ncaffe_out={}\ndiff={}'.format(pytorch_data, caffe_data, diff))
    print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' %
          (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))

    # save and print model_state
    # file1 = '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/pytorch_param.txt'
    # file2 = '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/caffe_param.txt'
    # save_model_state(file1, file2, pytorch_params, caffe_params)

