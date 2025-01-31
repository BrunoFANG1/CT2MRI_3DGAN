import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class StarGenerator3D(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(StarGenerator3D, self).__init__()

        layers = []
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose3d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm3d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv3d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        return self.model(x)
    

class StarDiscriminator3D(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=64, repeat_num=5):
        super(StarDiscriminator3D, self).__init__()
        layers = []
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = curr_dim * 2

        for i in range(2, repeat_num):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src
