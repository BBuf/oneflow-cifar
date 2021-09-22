'''Train CIFAR10 with OneFlow.'''
import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim
import oneflow.nn.functional as F
import oneflow.backends.cudnn as cudnn

# import oneflow.utils.vision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='OneFlow CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if flow.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = flow.utils.vision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = flow.utils.data.DataLoader(
#     trainset, batch_size=1024, shuffle=True, num_workers=2)

# testset = flow.utils.vision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = flow.utils.data.DataLoader(
#     testset, batch_size=1024, shuffle=False, num_workers=2)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
net.train()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = flow.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


class ResNet18TrainGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = net
        self.loss_fn = criterion
        self.add_optimizer(optimizer, lr_sch=scheduler)
    
    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss, y_pred

class ResNet18EvalGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = net
        self.loss_fn = criterion
    
    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

resnet18_train_graph = ResNet18TrainGraph()
resnet18_eval_graph = ResNet18EvalGraph()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (torch_inputs, torch_targets) in enumerate(trainloader):
        inputs = flow.tensor(torch_inputs.numpy(), requires_grad=False)
        targets = flow.tensor(torch_targets.numpy(), requires_grad=False)
        inputs, targets = inputs.to(device), targets.to(device)
        loss, outputs = resnet18_train_graph(inputs, targets)

        train_loss += loss.item()
        predicted = flow.argmax(outputs, 1).to(flow.int64)
        total += targets.size(0)

        correct += predicted.eq(targets).to(flow.int32).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with flow.no_grad():
        for batch_idx, (torch_inputs, torch_targets) in enumerate(testloader):
            inputs = flow.tensor(torch_inputs.numpy())
            targets = flow.tensor(torch_targets.numpy())
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = resnet18_eval_graph(inputs, targets)
            # loss = criterion(outputs, targets)

            test_loss += loss.item()
            # _, predicted = outputs.max(1)
            predicted = flow.argmax(outputs, 1).to(flow.int64)
            total += targets.size(0)

            
            correct += predicted.eq(targets).to(flow.int32).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
