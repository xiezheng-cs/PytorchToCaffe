import sys

sys.path.insert(0, '.')
import torch
import pytorch_to_caffe

from torch import nn
# from FaceFeatherNetB import FaceFeatherNetB
from FaceFeatherNetB_v2 import FaceFeatherNetB_v2


# def save_model_gpu2cpu(gpu_path, cpu_path):
#     net = FaceFeatherNetB_v2()
#     net = nn.DataParallel(net)
#     net = net.cuda()
#
#     checkpoint = torch.load(gpu_path)
#     state_dict = checkpoint['state_dict']
#     net.load_state_dict(state_dict)
#     print('model load gpu_state success!!')
#
#     check_point_params = {}
#     if isinstance(net, nn.DataParallel):
#         check_point_params["state_dict"] = net.module.state_dict()
#     else:
#         check_point_params["state_dict"] = net.state_dict()
#     torch.save(check_point_params, cpu_path)
#     print('save cpu_state success!!')


if __name__ == '__main__':

    # gpu_model to cpu_model
    # gpu_path = '/home/xiezheng/lcr/model/_34_best.pth.tar'
    # cpu_path = '/home/xiezheng/lcr/model/FaceFeatherNetB_baseline_34.pth'
    # save_model_gpu2cpu(gpu_path, cpu_path)

    # 186
    name = 'FaceFeatherNetB_se_prelu_checkpoint4'
    net = FaceFeatherNetB_v2(se=True)
    model_path = '/home/xiezheng/lcr/model/se_prelu_4_best.pth.tar'

    # name = 'FaceFeatherNetB_nose_prelu_checkpoint2'
    # net = FaceFeatherNetB_v2(se=False)

    checkpoint = torch.load(model_path)
    print(net)
    net.load_state_dict(checkpoint['model'])

    print('model_path={}'.format(model_path))
    print('load model success !!!')
    # assert False
    net.eval()
    input = torch.ones([1, 3, 112, 112])
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    print('model_path={}'.format(model_path))
    print('load model success !!!')