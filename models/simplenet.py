import oneflow as flow
import oneflow.nn as nn


class SimpleNet(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=8, padding=2),
            nn.BatchNorm2d(64),
        )
        self.linear = nn.Linear(1600, num_classes)


    def forward(self, x):
        conv_features = self.features(x)
        # print(conv_features.shape)
        conv_features = conv_features.view(conv_features.size(0), -1)
        # print(conv_features.shape)
        res = self.linear(conv_features)
        return res

net = SimpleNet()
x = flow.randn(1, 3, 32, 32)
y = net(x)


