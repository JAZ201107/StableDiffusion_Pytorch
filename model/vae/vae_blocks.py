import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x

        x = self.groupnorm(x)
        n, c, h, w = x.size()

        x = x.view((n, c, h * w)).transpose(-2, -1)
        x = self.attention(x).transpose(-2, -1).view((n, c, h, w))

        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.grroupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.grroupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residue = x

        x = self.grroupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.grroupnorm_2(x)
        x = F.silu(x)

        x = self.conv_2(x)
        return x + self.residual_layer(residue)
