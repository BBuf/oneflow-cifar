# Train CIFAR10 with OneFlow

I'm playing with [OneFlow](https://github.com/Oneflow-Inc/oneflow) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- OneFlow 0.5.0rc+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 93.92%|
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 95.62%|
| [ResNet50](https://arxiv.org/abs/1512.03385)          |       |
| [ResNet101](https://arxiv.org/abs/1512.03385)         |       |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     |       |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     |       |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       |       |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  |       |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  |       |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           |       |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       |       |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    |       |
| [DPN92](https://arxiv.org/abs/1707.01629)             |       |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           |       |

## Quantization Aware Training

If you are interested in OneFlow FX feature, please do the following to compile OneFlow Experience FX.

```
git clone https://github.com/Oneflow-Inc/oneflow
cd oneflow
git checkout add_fx_intermediate_representation
mkdir build
cd build
cmake -DCUDNN_ROOT_DIR=/usr/local/cudnn -DCMAKE_BUILD_TYPE=Release -DTHIRD_PARTY_MIRROR=aliyun -DUSE_CLANG_FORMAT=ON -DTREAT_WARNINGS_AS_ERRORS=OFF ..
make -j32
```

```
# Start training with: 
python main_qat.py

# You can manually resume the training with: 
python main_qat.py --resume --lr=0.01
```

Note:

The `momentum` parameter in the `MovingAverageMinMaxObserver` class defaults to 0.95, which will not be changed in the following experiments. 
## Accuracy
| Model             | quantization_bit | quantization_scheme | quantization_formula | per_layer_quantization | Acc |
| ----------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| ResNet18          |  8     |  symmetric      | google       |   True     |  95.19%      | 
| ResNet18          |  8     |  symmetric      | google       |   False    |  95.24%      | 
| ResNet18          |  8     |  affine         | google       |   True     |  95.32%      | 
| ResNet18          |  8     |  affine         | google       |   False    |  95.30%      | 
| ResNet18          |  8     |  symmetric      | cambricon    |   True     |  95.19%      |

## Reference
- https://github.com/kuangliu/pytorch-cifar

## TODO

add ddp train.

