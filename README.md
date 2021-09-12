# Train CIFAR10 with OneFlow

I'm playing with [OneFlow](https://github.com/Oneflow-Inc/oneflow) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- OneFlow 1.0+

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
| [VGG16](https://arxiv.org/abs/1409.1556)              |       |
| [ResNet18](https://arxiv.org/abs/1512.03385)          |       |
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

## TODO

add ddp.
