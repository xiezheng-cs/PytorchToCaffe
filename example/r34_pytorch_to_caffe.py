import sys

sys.path.insert(0, '.')
import torch
# from torch.autograd import Variable
# from torchvision.models import resnet
import pytorch_to_caffe

from insightface_resnet_pruned import pruned_LResNet34E_IR



if __name__ == '__main__':
    # name = 'resent34_baseline'
    # net = pruned_LResNet34E_IR(pruning_rate=0.0)
    # model_path = '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/log/insightface_r34/insightface_r34_with_arcface_epoch48.pth'

    name = 'resent34_pruned0.25'
    net = pruned_LResNet34E_IR(pruning_rate=0.25)
    model_path = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_r34/pytorch_pruned0.25_r34_lr0.01_checkpoint26_feature-result_fliped/checkpoint_026.pth'

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