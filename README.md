
## Pytorch to Caffe

The new version of pytorch_to_caffe supporting the newest version(from 0.2.0 to 0.4.1) of pytorch.
NOTICE: The transfer output will be somewhat different with the original model, caused by implementation difference.

- Supporting layers types:
```angular2html
conv2d  ->  Convolution, 
_conv_transpose2d ->  Deconvolution, 
_linear -> InnerProduct, 
_split  -> Slice, 
max_pool2d,_avg_pool2d   -> Pooling,
_max    -> Eltwise, 
_cat    -> Concat,
dropout -> Dropout,
 relu   -> ReLU, 
 prelu  -> PReLU, 
 _leaky_relu -> ReLU,
 _tanh  -> TanH,   
 threshold(only value=0) -> Threshold,ReLU,
 softmax -> Softmax, 
 batch_norm -> BatchNorm,Scale, 
 instance_norm -> BatchNorm,Scale,
 _interpolate  ->  Upsample
 ```

- Supporting operations: torch.split, torch.max, torch.cat
- Supporting tensor Variable operations: var.view, + (add), += (iadd), -(sub), -=(isub), * (mul), *= (imul)

Need to be added for caffe in the future:
- Normalize,DepthwiseConv

The supported above can transfer many kinds of nets, 
such as AlexNet(tested), VGG(tested), ResNet(fixed the bug in origin repo which mainly caused by ReLu layer function.), Inception_V3(tested).

The supported layers concluded the most popular layers and operations.
 The other layer types will be added soon, you can ask me to add them in issues.

Example: please see file `example/alexnet_pytorch_to_caffe.py`. Just Run `python3 example/alexnet_pytorch_to_caffe.py`

## Deploy verify
After Converter,we should use verify_deploy.py to verify the output of pytorch model and the convertted caffe model.
If you want to verify the outputs of caffe and pytorch,you should make caffe and pytorch install in the same environment,anaconda is recommended.
using following script,we can install caffe-gpu(master branch). 
```angular2html
conda install caffe-gpu pytorch cudatoolkit=9.0 -c pytorch 

```
please see file `example/verify_deploy.py`,it can verify the output of pytorch model and the convertted caffe model in the same input.


## #####################################################################################################################


## Add something
 - Supporting layers types:
```angular2html
        nn.AdaptiveAvgPool2d
        nn.Sigmoid
```

## A method to convert SELayer from pytorch to caffe
```angular2html
layer {
  name: "view2"
  type: "Reshape"
  bottom: "Sigmoid_blob1"
  top: "view_blob2"
  reshape_param {
    shape {
      dim: 0
      dim: 16
      dim: 1
      dim: 1
    }
  }
}
layer {
  name: "mul1"
  type: "Eltwise"
  bottom: "batch_norm_blob3"
  bottom: "view_blob2"
  top: "mul_blob1"
  eltwise_param {
    operation: PROD
  }
}
```
edit to
```angular2html
layer {
  name: "view2"
  type: "Reshape"
  bottom: "Sigmoid_blob1"
  top: "view_blob2"
  reshape_param {
    shape {
      dim: 0
      dim: 16
    }
  }
}
layer {
  name: "mul1"
  type: "Scale"
  bottom: "batch_norm_blob3"
  bottom: "view_blob2"
  top: "mul_blob1"
  scale_param {
    axis: 0
    bias_term: false
  }
}
```

## NCNN-Quantization
This convert tools is base on TensorRT 2.0 Int8 calibration tools,which use the KL algorithm to find the suitable threshold to quantize the activions from Float32 to Int8(-128 - 127).
```angular2html
python caffe-int8-convert-tool-dev-weight.py -h
usage: caffe-int8-convert-tool-dev-weight.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--group GROUP] [--gpu GPU]

find the pretrained caffemodel int8 quantize scale value
```
```angular2html
optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained caffemodel
  --mean MEAN           value of mean
  --norm NORM           value of normalize(scale value)
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --group GROUP         enable the group scale(0:disable,1:enable,default:1)
  --gpu GPU             use gpu to forward(0:disable,1:enable,default:0)
python caffe-int8-convert-tool-dev-weight.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 127.5 127.5 127.5 --norm=0.0078125 --images=test/images/ -output=mobilenet_v1.table --group=1 --gpu=1
```


