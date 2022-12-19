import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, upscale_factor, in_channels, out_channels, activation=True):
        super(ESPCN, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, out_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        if self.activation:
            x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        else:
            x = self.pixel_shuffle(self.conv3(x))
        return x