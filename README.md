# model-compression-and-acceleration-progress
Repository to track the progress in model compression and acceleration

## Low-rank approximation

- Paper 1 \
| paper | code | dataset : model | metrics

## Pruning

- AMC: AutoML for Model Compression and Acceleration on Mobile Devices (ECCV18)
[paper](https://arxiv.org/abs/1802.03494) | [pretrained models (TensorFlow, TensorFlow Light)](https://github.com/mit-han-lab/amc-compressed-models)
- ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression (ICCV 2017)
[paper](https://arxiv.org/abs/1707.06342) | [pretrained model (Caffe)](https://github.com/Roll920/ThiNet)
- SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR 2017)
[paper](https://arxiv.org/abs/1704.08063) | [code and pretrained models (Caffe)](https://github.com/isthatyoung/Sphereface-prune) 

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
- Knowledge disstillation + quantization (Pytorch)
https://github.com/antspy/quantized_distillation

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
