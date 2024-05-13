import torch
import torch.nn as nn

from src.modeling.base import BaseModel

'''
K - kernel size
P - padding
S - stride
w_out = (w_in - K + 2P) // S + 1 
'''


class SpatiallySeparableConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            padding,
    ):
        super(SpatiallySeparableConvBlock, self).__init__()
        self.conv_1xn = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.conv_nx1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_1xn(x)
        x = self.conv_nx1(x)
        x = self.activation(x)
        x = self.pool(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            padding,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)

        return x


class DDulaevAutoEncoder(BaseModel):
    def __init__(self, model_name='ddulaev_auto_encoder'):
        super(DDulaevAutoEncoder, self).__init__(model_name)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=128, kernel_size=7, padding=3),
            SpatiallySeparableConvBlock(in_channels=128, out_channels=32, kernel_size=5, padding=2),
            ConvBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x
