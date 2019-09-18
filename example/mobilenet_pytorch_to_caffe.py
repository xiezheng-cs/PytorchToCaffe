import sys

sys.path.insert(0, '.')
import torch
# from torch.autograd import Variable
# from torchvision.models import resnet
import pytorch_to_caffe

from pruned_mobilenetv1 import PrunedMobileNetV1

if __name__ == '__main__':
    name = 'imagenet_mobilenetv1'
    net = PrunedMobileNetV1(pruning_rate=0.0)
    model_path = ''

    checkpoint = torch.load(model_path)
    print(net)
    net.load_state_dict(checkpoint['model'])
    print('model_path={}'.format(model_path))
    print('load model success !!!')
    # assert False
    net.eval()
    # input = torch.ones([1, 3, 112, 112])
    input = torch.ones([1, 3, 224, 224])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    print('model_path={}'.format(model_path))
    print('load model success !!!')
