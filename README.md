# Train CIFAR10 with OneFlow Cambricon in MLU270

I'm playing with [oneflow-cambricon](https://github.com/Oneflow-Inc/oneflow-cambricon) on the CIFAR10 dataset.

## Prerequisites
- Python 3.7+
- OneFlow cambricon

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
| [ResNet50](https://arxiv.org/abs/1512.03385)          |  |
## Reference
- https://github.com/kuangliu/pytorch-cifar

## TODO

add ddp train.

