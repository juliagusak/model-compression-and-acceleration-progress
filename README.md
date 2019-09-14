# Model Compression and Acceleration Progress
Repository to track the progress in model compression and acceleration

## Low-rank approximation

- MUSCO: Multi-Stage COmpression of neural networks (2019)
[paper](https://arxiv.org/abs/1903.09973) | [code (PyTorch)](https://github.com/juliagusak/musco)
- Efficient Neural Network Compression (CVPR 2019)
[paper](https://arxiv.org/abs/1811.12781) | [code (Caffe)](https://github.com/Hyeji-Kim/ENC) 
- Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling (ICLR 2019)
[paper](https://openreview.net/pdf?id=B1eHgu-Fim) | [code (PyTorch)](https://github.com/zuenko/ALRF)
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


## Pruning & Sparsification
#### Papers
- Rethinking the Value of Network Pruning (ICLR 2019, NIPS 2018 workshop) 
[paper](https://arxiv.org/abs/1810.05270) | [code (PyTorch)](https://github.com/Eric-mingjie/rethinking-network-pruning)
- Dynamic Channel Pruning: Feature Boosting and Suppression (ICLR 2019)
[paper](https://arxiv.org/abs/1810.05331) | [code](https://github.com/deep-fry/mayo)
- AutoPruner: An End-to-End Trainable Filter Pruning Method for Efficient Deep Model Inference (2019)
[paper](https://arxiv.org/abs/1805.08941)
- CLIP-Q: Deep Network Compression Learning by In-ParallelPruning-Quantization (CVPR 2018)
[paper](http://www.sfu.ca/~ftung/papers/clipq_cvpr18.pdf)
- Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks (IJCAI 2018)
[paper](https://arxiv.org/abs/1808.06866) | [code and models (PyTorch)](https://github.com/he-y/soft-filter-pruning)
- Discrimination-aware Channel Pruning for Deep Neural Networks (NIPS 2018)
[paper](https://papers.nips.cc/paper/7367-discrimination-aware-channel-pruning-for-deep-neural-networks.pdf) | [code and pretrained models (PyTorch)](https://github.com/SCUT-AILab/DCP)
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices (ECCV18)
[paper](https://arxiv.org/abs/1802.03494) | [code (PyTorch)](https://github.com/mit-han-lab/amc-release) | [pretrained models (PyTorch, TensorFlow, TensorFlow Light)](https://github.com/mit-han-lab/amc-compressed-models)
- Channel Gating Neural Networks (2018)
[paper](https://arxiv.org/abs/1805.12549)
- DSD: Dense-Sparse-Dense Training for Deep Neural Networks [paper](https://arxiv.org/abs/1607.04381) | [pretrained models (Caffe)](https://songhan.github.io/DSD/) (ICLR 2017)
- Channel Pruning for Accelerating Very Deep Neural Networks (ICCV 2017)
[paper](https://arxiv.org/abs/1707.06168) | [code and pretrained models (Caffe)](https://github.com/yihui-he/channel-pruning) | [code (PyTorch)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet)
- Learning Efficient Convolutional Networks through Network Slimming (ICCV 2017)
[paper](https://arxiv.org/abs/1708.06519) | [code (Torch, Pytorch)](https://github.com/Eric-mingjie/network-slimming)
- ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression (ICCV 2017)
[paper](https://arxiv.org/abs/1707.06342) | [pretrained model (Caffe)](https://github.com/Roll920/ThiNet) | [code (PyTorch)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet)
- Structured Bayesian Pruning via Log-Normal Multiplicative Noise (NIPS 2017)
[paper](https://papers.nips.cc/paper/7254-structured-bayesian-pruning-via-log-normal-multiplicative-noise.pdf) | [code (TensorFlow, Theano + Lasagne)](https://github.com/necludov/group-sparsity-sbp)
- SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR 2017)
[paper](https://arxiv.org/abs/1704.08063) | [code and pretrained models (Caffe)](https://github.com/isthatyoung/Sphereface-prune) 
- Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding (ICLR 2016)
[paper](https://arxiv.org/abs/1510.00149)
- Fast ConvNets Using Group-wise Brain Damage (CVPR 2016)
[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Lebedev_Fast_ConvNets_Using_CVPR_2016_paper.pdf)

#### Repos
- Pruning + quantization [code and pretrained models (TensorFlow, TensorFlow light)](https://github.com/vikranth94/Model-Compression). Examples for CIFAR.


## Knowledge distillation 
#### Papers
- Learning Efficient Detector with Semi-supervised Adaptive Distillation (arxiv 2019) [paper](https://arxiv.org/abs/1901.00366) | [code (Caffe)](https://github.com/Tangshitao/Semi-supervised-Adaptive-Distillation)
- Model compression via distillation and quantization (ICLR 2018) [paper](https://arxiv.org/abs/1802.05668) | [code (Pytorch)](https://github.com/antspy/quantized_distillation)
- Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks (ICLR 2018 workshop)
[paper](https://arxiv.org/abs/1709.00513)
- Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks ( BMVC 2018)
[paper](https://arxiv.org/abs/1709.00513)
- Net2Net: Accelerating Learning via Knowledge Transfer (ICLR 2016)
[paper](https://arxiv.org/abs/1511.05641)
- Distilling the Knowledge in a Neural Network (NIPS 2014)
[paper](https://arxiv.org/abs/1503.02531)
- FitNets: Hints for Thin Deep Nets (2014)
[paper](https://arxiv.org/abs/1412.6550) | [code (Theano + Pylearn2)](https://github.com/adri-romsor/FitNets)

#### Repos
TensorFlow implementation of three papers https://github.com/chengshengchan/model_compression, results for CIFAR-10

## Quantization
- TODO

## Architecture search
- MobileNets
  - Searching for MobileNetV3
  [paper](https://arxiv.org/abs/1905.02244)
  - MobileNetV2: Inverted Residuals and Linear Bottlenecks (CVPR 2018)
  [paper](https://arxiv.org/abs/1801.04381) | [code and pretrained models (TensorFlow)](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019)
[paper](https://arxiv.org/abs/1905.11946) | [code and pretrained models (TensorFlow)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- MnasNet: Platform-Aware Neural Architecture Search for Mobile (CVPR 2019)
[paper](https://arxiv.org/abs/1807.11626) | [code (TensorFlow)](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)
- MorphNet: Fast & Simple Resource-Constrained Learning of Deep Network Structure (CVPR 2018) 
[paper](https://arxiv.org/abs/1711.06798) | [code (TensorFlow)](https://github.com/google-research/morph-net)
- ShuffleNets
  - ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design (ECCV 2018)
  [paper](https://arxiv.org/abs/1807.11164)
  - ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (CVPR 2018)
  [paper](https://arxiv.org/abs/1707.01083)
- Multi-Fiber Networks for Video Recognition (ECCV 2018)
[paper](https://arxiv.org/abs/1807.11195) | [code (PyTorch)](https://github.com/cypw/PyTorch-MFNet)
- IGCVs
  - IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks (BMVC 2018)
  [paper](https://arxiv.org/abs/1806.00178) | [code and pretrained models (MXNet)](https://github.com/homles11/IGCV3)
  - IGCV2: Interleaved Structured Sparse Convolutional Neural Networks (CVPR 2018)
  [paper](https://arxiv.org/abs/1804.06202)
  - Interleaved Group Convolutions for Deep Neural Networks (ICCV 2017)
  [paper](https://arxiv.org/abs/1707.02725)


## PhD thesis and overviews

- Algorithms for speeding up convolutional neural networks (2018) [thesis](https://www.skoltech.ru/app/data/uploads/2018/10/Thesis-Final.pdf)
- Model Compression and Acceleration for Deep Neural Networks: The Principles, Progress, and Challenges (2018) [paper](http://cwww.ee.nctu.edu.tw/~cfung/docs/learning/cheng2018DNN_model_compression_accel.pdf)
- Efficient methods and hardware for deep learning (2017) [thesis](https://stacks.stanford.edu/file/druid:qf934gh3708/EFFICIENT%20METHODS%20AND%20HARDWARE%20FOR%20DEEP%20LEARNING-augmented.pdf)


## Frameworks
- [MUSCO](https://github.com/juliagusak/musco) - framework for model compression using tensor decompositions (PyTorch)
- [Distiller](https://github.com/NervanaSystems/distiller) - package for compression using pruning and low-precision arithmetic (PyTorch)
- [MorphNet](https://github.com/google-research/morph-net) - framework for neural networks architecture learning (TensorFlow)
- [Mayo](https://github.com/deep-fry/mayo) - deep learning framework with fine- and coarse-grained pruning, network slimming, and quantization methods 
- [PocketFlow](https://github.com/Tencent/PocketFlow) - framework for model pruning, sparcification, quantization (TensorFlow implementation) 
- [Keras compressor](https://github.com/DwangoMediaVillage/keras_compressor) - compression using low-rank approximations, SVD for matrices, Tucker for tensors.
- [Caffe compressor](https://github.com/yuanyuanli85/CaffeModelCompression) K-means based quantization



## Comparison of different approaches

Please, see ```comparative_results.pdf``` 


#### 

## Similar repos

- https://github.com/ZhishengWang/Embedded-Neural-Network
- https://github.com/memoiry/Awesome-model-compression-and-acceleration
- https://github.com/sun254/awesome-model-compression-and-acceleration
- https://github.com/guan-yuan/awesome-AutoML-and-Lightweight-Models
- https://github.com/chester256/Model-Compression-Papers
- https://github.com/mapleam/model-compression-and-acceleration-4-DNN
- https://github.com/cedrickchee/awesome-ml-model-compression
- https://github.com/jnjaby/Model-Compression-Acceleration
