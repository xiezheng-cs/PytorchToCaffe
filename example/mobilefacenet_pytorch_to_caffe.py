import sys
import os

sys.path.insert(0, '.')
import torch
# from torch.autograd import Variable
# from torchvision.models import resnet
import pytorch_to_caffe
# from MobileNetV2 import MobileNetV2

from mobilefacenet_pruned import pruned_Mobilefacenet
from models import MobileFaceNet
from mobilefacenetv2_width_wm import Mobilefacenetv2_width_wm


if __name__ == '__main__':
    # name = 'ms1mv2_mobilefacenet_p0.25'
    # net = pruned_Mobilefacenet(pruning_rate=0.25)
    # model_path = '/home/dataset/xz_datasets/Megaface/ms1mv2_pytorch_mobilefacenet/pytorch_pruned0.25_mobilefacenet_v1_better/p0.25_mobilefacenet_v1_without_fc_checkpoint_022.pth'

    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_17_without_p_fc_last1_1420'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_nir_finetune_sq_without_p_fc/log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1420_align_bs200_e30_lr0.010_step[10, 20]_pruned0.5_nir_finetune_last_layer_17_20190827/check_point/checkpoint_029.pth'

    # name = 'iccv_mobilefacenet_p0.5_w_a_int8-finetune_17_without_p_fc_last2_1420'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_nir_finetune_sq_without_p_fc/log_aux_mobilefacenetv2_baseline_0.5width_128_arcface_jidian_nir_1420_align_bs200_e30_lr0.010_step[10, 20]_pruned0.5_nir_finetune_last_two_layer_20190827_17/check_point/checkpoint_029.pth'

    # qqc_sq
    # name = 'iccv_mobilefacenet_p0.5_without_p_fc_our_sq_then_soft_tinetune_qqc_sq_1224'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_our_sq_then_nir_soft_finetune_1224_epoch26_qqc_sq/check_point/checkpoint_25.pth'

    # name = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_step[10,20]/iccv_mobilefacenet_p0.5_without_p_fc_qqc_sq_chp11_then_soft_tinetune_qqc_sq_1224_chp25_step10-20'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_step[10,20]/check_point/checkpoint_25.pth'

    # name = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_cos_lr/iccv_mobilefacenet_p0.5_without_p_fc_qqc_sq_chp11_then_soft_tinetune_qqc_sq_1224_chp25_cos_lr'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_cos_lr/check_point/checkpoint_25.pth'

    # name = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_nostep/iccv_mobilefacenet_p0.5_without_p_fc_qqc_sq_chp11_then_soft_tinetune_qqc_sq_1224_chp25_nostep'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1224_epoch26_qqc_sq_nostep/check_point/checkpoint_25.pth'

    # name = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1321_epoch26_qqc_sq_nostep/iccv_mobilefacenet_p0.5_without_p_fc_qqc_sq_chp11_then_soft_tinetune_qqc_sq_1321_chp25_nostep'
    # net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    # model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1321_epoch26_qqc_sq_nostep/check_point/checkpoint_25.pth'

    name = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1420_epoch26_qqc_sq_nostep/iccv_mobilefacenet_p0.5_without_p_fc_qqc_sq_chp11_then_soft_tinetune_qqc_sq_1420_chp25_nostep'
    net = Mobilefacenetv2_width_wm(embedding_size=128, pruning_rate=0.5)
    model_path = '/home/liujing/NFS/ICCV_challenge/xz_log/iccv_emore_log_gdc/baseline/mobilefacenet/2stage/dim128/mf_p0.5_without_p_fc_nir_soft_finetune_lr0.0001/mf_p0.5_without_p_fc_qqc_sq_ckp11_then_nir_soft_finetune_1420_epoch26_qqc_sq_nostep/check_point/checkpoint_25.pth'

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