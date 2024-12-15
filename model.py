import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            stride=stride,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Strided convolution with dilation
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2,
                     padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Depthwise separable convolution with stride
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Final strided convolution
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Global Average Pooling and final FC layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # 32x32 -> 32x32
        x = self.conv2(x)  # 32x32 -> 16x16
        x = self.conv3(x)  # 16x16 -> 8x8
        x = self.conv4(x)  # 8x8 -> 4x4
        x = self.gap(x)    # 4x4 -> 1x1
        x = x.view(-1, 128)
        x = self.fc(x)
        return x 