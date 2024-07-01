import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_size: int, hidden_size: int, out_size: int, pad: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=3, padding=pad)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=3, padding=pad)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

        if out_size == in_size:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_size, out_size, kernel_size=1)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)), inplace=True)
        x = F.relu(self.batchnorm2(self.conv2(x)), inplace=True)
        return x

    def forward(self, x):
        return self.skip_connection(x) + self.convblock(x)  # skip connection


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_size=in_ch, hidden_size=out_ch, out_size=out_ch),
            ResBlock(out_ch, hidden_size=out_ch, out_size=out_ch),
            CAModule(in_channels=out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bicubic=True):
        super().__init__()
        self.upsample = None
        if bicubic:
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode="bicubic",
                align_corners=True,
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch // 2, kernel_size=2, stride=2
            )
        self.conv = nn.Sequential(
            ResBlock(in_ch, hidden_size=out_ch, out_size=out_ch),
            ResBlock(out_ch, hidden_size=out_ch, out_size=out_ch),
        )
        # self.conv = DoubleConv(in_ch, out_ch)
        self.bilinear_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        bilinear_output = self.bilinear_up(x1)
        return self.conv(x), bilinear_output


class CAModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
