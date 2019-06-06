# model-compression-and-acceleration-progress
Repository to track the progress in model compression and acceleration

## Low-rank approximation

- Efficient Neural Network Compression (CVPR 2019)
[paper](https://arxiv.org/abs/1811.12781) | [code (Caffe)](https://github.com/Hyeji-Kim/ENC) 
- Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling (ICLR 2019)
[paper](https://openreview.net/pdf?id=B1eHgu-Fim) | [code (PyTorch)](https://github.com/zuenko/ALRF)
- MUSCO: Multi-Stage COmpression of neural networks (2019)
[paper](https://arxiv.org/abs/1903.09973) | [code (PyTorch)](https://github.com/juliagusak/musco)
- Extreme Network Compression via Filter Group Approximation (ECCV 2018)
[paper](https://arxiv.org/abs/1807.11254)
- Ultimate tensorization: compressing convolutional and FC layers alike (NIPS 2016 workshop)
[paper](https://arxiv.org/abs/1611.03214) | [code (TensorFlow)](https://github.com/timgaripov/TensorNet-TF) | [code (MATLAB, Theano + Lasagne)](https://github.com/Bihaqo/TensorNet)
- Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications (ICLR 2016)
[paper](https://arxiv.org/abs/1511.06530) 
- Accelerating Very Deep Convolutional Networks for Classification and Detection (IEEE TPAMI 2016)
[paper](https://arxiv.org/abs/1505.06798)
- Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition (ICLR 2015)
[paper](https://arxiv.org/abs/1412.6553) | [code (Caffe)](https://github.com/vadim-v-lebedev/cp-decomposition)
- Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation (NIPS 2014)
[paper](https://arxiv.org/abs/1404.0736)
- Speeding up Convolutional Neural Networks with Low Rank Expansions (2014)
[paper](https://arxiv.org/abs/1405.3866)




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
- Model compression via distillation and quantization (ICLR 2018) [paper](https://arxiv.org/abs/1802.05668) | [code (Pytorch)](https://github.com/antspy/quantized_distillation)
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
