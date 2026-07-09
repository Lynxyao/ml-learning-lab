# -*- coding: utf-8 -*-
"""
PyTorch 版 WFM 网络结构
——————————————
对应原来的 network.py（TF1）
实现：
 - U-Net 生成器（Generator）
 - PatchGAN 判别器（Discriminator）

输入通道：3（原图是 RGB）
输出通道：3（力场灰度图也按 3 通道保存，和原代码兼容）
"""

import torch
import torch.nn as nn

# ---------- 基本模块 ----------

class UNetDown(nn.Module):
    """ U-Net 下采样块：Conv2d -> (BatchNorm) -> LeakyReLU """
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ U-Net 上采样块：ConvTranspose2d -> BatchNorm -> ReLU -> (Dropout) -> concat skip """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        # 与编码器对应层做 skip connection
        x = torch.cat((x, skip), dim=1)
        return x


# ---------- 生成器：U-Net ----------

class GeneratorUNet(nn.Module):
    """
    生成器 G：
        输入： (N, 3, 256, 256) 皱纹/显微镜图
        输出： (N, 3, 256, 256) 力场灰度图
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 编码器 8 层（和原 U-Net 结构类似）
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 128x128
        self.down2 = UNetDown(64, 128)                           # 64x64
        self.down3 = UNetDown(128, 256)                          # 32x32
        self.down4 = UNetDown(256, 512)                          # 16x16
        self.down5 = UNetDown(512, 512)                          # 8x8
        self.down6 = UNetDown(512, 512)                          # 4x4
        self.down7 = UNetDown(512, 512)                          # 2x2
        self.down8 = UNetDown(512, 512, normalize=False)         # 1x1

        # 解码器 7 个上采样块 + 最后一层输出
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()  # 输出范围 [-1, 1]，方便和 util.norm() 对应
        )

    def forward(self, x):
        # 编码
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # 解码 + skip
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        out = self.final(u7)
        return out


# ---------- 判别器：PatchGAN ----------

class Discriminator(nn.Module):
    """
    PatchGAN 判别器 D：
        输入： (x, y) 拼在一起 (N, 6, 256, 256)
        输出： (N, 1, H', W') patch 真伪评分
    """
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 输入通道 = input(3) + target(3) = 6
        self.model = nn.Sequential(
            *block(in_channels * 2, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)  # 不加 sigmoid，用 BCEWithLogitsLoss
        )

    def forward(self, x, y):
        # 通道维拼接
        inp = torch.cat((x, y), dim=1)
        return self.model(inp)
