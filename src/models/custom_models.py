import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv(3, 6, 5),  # -> 6 x 110 * 110
            Conv(6, 16, 3),  # -> 16 x 54 * 54
            Conv(16, 120, 5),  # -> 120 x 25 * 25
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(120 * 25 * 25, 84),
            nn.ReLU(),
            nn.Linear(84, 42),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # -> (227-11)/4 + 1 = 55
            nn.MaxPool2d(3, stride=2),  # -> (55-3)/2 + 1 = 27
            nn.ReLU(),
            nn.BatchNorm2d(96),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # -> (27+2*2-5)/1 + 1 = 27
            nn.MaxPool2d(3, stride=2),  # -> (27-3)/2 + 1 = 13
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # -> (13+2*1-3)/1 + 1 = 13
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # -> (13+2*1-3)/1 + 1 = 13
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # -> (13 + 2*1 -3)/1 + 1 = 13
            nn.MaxPool2d(3, stride=2),  # -> (13 - 3)/2 + 1 = 6
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4096, 42),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
