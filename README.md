# model-compression-and-acceleration-progress
Repository to track the progress in model compression and acceleration

## Low-rank approximation

- Paper 1 \
| paper | code | dataset : model | metrics

## Pruning
### Papers
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices (ECCV18)
[paper](https://arxiv.org/abs/1802.03494) | [pretrained models (TensorFlow, TensorFlow Light)](https://github.com/mit-han-lab/amc-compressed-models)
- ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression (ICCV 2017)
[paper](https://arxiv.org/abs/1707.06342) | [pretrained model (Caffe)](https://github.com/Roll920/ThiNet)
- SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR 2017)
[paper](https://arxiv.org/abs/1704.08063) | [code and pretrained models (Caffe)](https://github.com/isthatyoung/Sphereface-prune) 

### Repos
- Pruning + quantization [code and pretrained models (TensorFlow, TensorFlow light)](https://github.com/vikranth94/Model-Compression). Examples for CIFAR.

## Quantization

- Paper 3 \
| paper | code | dataset : model | metrics

## Optimal architecture search 
- Paper 5 \
| paper | code | dataset : model | metrics

## Knowledge distillation 

### Papers
- Paper 4 \
| paper | code | dataset : model | metrics
- Knowledge disstillation + quantization [code (Pytorch)](https://github.com/antspy/quantized_distillation)
- Learning Efficient Detector with Semi-supervised Adaptive Distillation (arxiv 2019) [paper](https://arxiv.org/abs/1901.00366) | [code (Caffe)](https://github.com/Tangshitao/Semi-supervised-Adaptive-Distillation)

### Repos
TensorFlow implementation of three papers https://github.com/chengshengchan/model_compression, results for CIFAR-10


## Frameworks
- [PocketFlow](https://github.com/Tencent/PocketFlow) - framework for model pruning, sparcification, quantization (TensorFlow implementation) 
- [Keras compressor](https://github.com/DwangoMediaVillage/keras_compressor) - compression using low-rank approximations, SVD for matrices, Tucker for tensors.
- [Caffe compressor](https://github.com/yuanyuanli85/CaffeModelCompression) K-means based quantization

## Similar repos

- https://github.com/ZhishengWang/Embedded-Neural-Network
- https://github.com/memoiry/Awesome-model-compression-and-acceleration
- https://github.com/sun254/awesome-model-compression-and-acceleration
- https://github.com/guan-yuan/awesome-AutoML-and-Lightweight-Models
- https://github.com/chester256/Model-Compression-Papers
- https://github.com/mapleam/model-compression-and-acceleration-4-DNN
- https://github.com/cedrickchee/awesome-ml-model-compression
- https://github.com/jnjaby/Model-Compression-Acceleration
