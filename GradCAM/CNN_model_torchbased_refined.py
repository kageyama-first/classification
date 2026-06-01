import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        k = int(kernel_size)
        if k not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels=channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class SimpleCNN(nn.Module):
    """无 BatchNorm，无 Dropout，网络较浅"""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 输入尺寸: 3 x 224 x 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 112 x 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 28 x 28
            nn.AdaptiveAvgPool2d((7, 7)) # 强制统一下一步尺寸为 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class SimpleCNN_CBAM(nn.Module):
    def __init__(self, num_classes, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.cbam1 = CBAM(16, reduction=reduction, spatial_kernel_size=spatial_kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.cbam2 = CBAM(32, reduction=reduction, spatial_kernel_size=spatial_kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.cbam3 = CBAM(64, reduction=reduction, spatial_kernel_size=spatial_kernel_size)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.cbam1(self.relu1(self.conv1(x))))
        x = self.pool2(self.cbam2(self.relu2(self.conv2(x))))
        x = self.pool3(self.cbam3(self.relu3(self.conv3(x))))
        x = self.avgpool(x)
        return self.classifier(x)


class AdvancedCNN(nn.Module):
    """引入 BatchNorm、Dropout，增加网络深度和通道数"""
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #32 x 112 x 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #64 x 56 x 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #128 x 28 x 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #256 x 14 x 14
            
            nn.AdaptiveAvgPool2d((4, 4)) #256 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
