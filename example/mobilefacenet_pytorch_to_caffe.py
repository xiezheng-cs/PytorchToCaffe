import sys

sys.path.insert(0, '.')
import torch
# from torch.autograd import Variable
# from torchvision.models import resnet
import pytorch_to_caffe
# from MobileNetV2 import MobileNetV2
from mobilefacenet_pruned import pruned_Mobilefacenet
from insightface_resnet_pruned import pruned_LResNet34E_IR
from models import MobileFaceNet
from mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm


if __name__ == '__main__':

    # name = 'MobileNetV2'
    # net= MobileNetV2()
    # checkpoint = torch.load("/home/shining/Downloads/mobilenet_v2.pth.tar")

    # 186
    # name = 'mobilefacenet'
    # net = MobileFaceNet(embedding_size=128, blocks = [1,4,6,2])
    # checkpoint = torch.load('/home/dataset/xz_datasets/jidian_face_499/model/mobilefacenet_v1_epoch34.pth')

    # name = 'mobilefacenet_p0.5_w_a_int8-finetune_017'
    # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # origin_fintune
    # checkpoint = torch.load('/home/dataset/xz_datasets/jidian_face_499/model/'
    #                      'mobilefacenet_cos_lr_pruned_0.5_nowith_fc_checkpoint_026.pth')
    # # w_int8_finetune
    # checkpoint = torch.load('/home/dataset/xz_datasets/jidian_face_499/model/'
    #                         'mobilefacenet_cos_lr_pruned_0.5_nowith_fc_int8_finetune_checkpoint_027.pth')
    # w_a_int8_finetune
    # checkpoint = torch.load('/home/dataset/xz_datasets/jidian_face_9784/'
    #                         'pytorch_mobilefacenet_cos_lr_p0.5_nowith_fc_w_a_int8_finetune_checkpoint_017.pth')

    # name = 'mobilefacenet_p0.5_w_a_int8-finetune_nir_last3_029'
    # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last3
    # checkpoint = torch.load('/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/'
    #                         'pytorch_model/pytroch_mobilefacenet_p0.5_w_a_int8-finetune_nir_last3_checkpoint_29.pth')

    # name = 'mobilefacenet_p0.5_w_a_int8-finetune_nir_last2_029'
    # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last2
    # checkpoint = torch.load(
    #     '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/'
    #     'pytorch_model/pytroch_mobilefacenet_p0.5_w_a_int8-finetune_nir_last2_checkpoint_29.pth')

    # name = 'mobilefacenet_p0.5_w_a_int8-finetune_nir_last1_029'
    # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last1
    # model_path = '/home/xiezheng/programs2019/insightface_DCP/insightface_v2/nir_face_dataset/PytorchToCaffe/pytorch_model/' \
    #              'pytroch_mobilefacenet_p0.5_w_a_int8-finetune_nir_last1_checkpoint_29.pth'
    # checkpoint = torch.load(model_path)


    # # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_017_liujing'
    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_016_xz'
    #
    # # net = pruned_Mobilefacenet(pruning_rate=0.5)
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # # w_a_int8_finetune_xz
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/quantization/' \
    #              'log_ft_mobilefacenet_v1_p0.5_iccv_ms1m_bs256_e18_lr0.001_step[]_new_mf_p0.5_cos_lr_w_a_int8' \
    #              '_finetune_20190806_wd4e-5/check_point/checkpoint_016.pth'
    #
    # # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/quantization/' \
    # #              'log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_iccv_emore_bs160_e18_lr0.001_step[]_cosine_quantization_' \
    # #              'finetune_20190806_p0.5/check_point/checkpoint_017.pth'
    # checkpoint = torch.load(model_path)

    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_nir_last1_029'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last1
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/nir_finetune_sq/' \
    #              'log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1224_bs200_e30_lr0.010_step[10, 20]' \
    #              '_pruned0.5_nir_finetune_last_layer_20190809/check_point/checkpoint_029.pth'
    # checkpoint = torch.load(model_path)

    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_nir_last2_029'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last2
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/nir_finetune_sq/' \
    #              'log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1224_bs200_e30_lr0.010_step[10, 20]' \
    #              '_pruned0.5_nir_finetune_last_two_layer_20190809/check_point/checkpoint_029.pth'
    # checkpoint = torch.load(model_path)

    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_nir_last3_029'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # # w_a_int8_finetune_nir_last3
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/nir_finetune_sq/' \
    #              'log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1224_bs200_e30_lr0.010_step[10, 20]' \
    #              '_pruned0.5_nir_finetune_last_three_layer_20190809/check_point/checkpoint_029.pth'


    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_12_without_p_fc'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/' \
    #              'quantization/log_aux_mobilefacenetv2_baseline_0.5width_without_fc_128_arcface_iccv_emore_bs384' \
    #              '_e18_lr0.001_step[]_without_fc_cosine_quantization_finetune_20190811/check_point/checkpoint_012.pth'

    name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_09_without_p_fc_last2'
    net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_nir_finetune_sq_without_p_fc/log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1224_bs200_e30_lr0.010_step[10, 20]_pruned0.5_nir_finetune_last_two_layer_20190813_09/check_point/checkpoint_029.pth'
    checkpoint = torch.load(model_path)

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