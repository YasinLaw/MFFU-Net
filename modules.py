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
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bicubic=True, fusion_factor=2):
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
        self.bypass_fusion = nn.Upsample(
            scale_factor=fusion_factor, mode="bicubic", align_corners=True
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x), self.bypass_fusion(x1)


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


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoorAttn(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoorAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
