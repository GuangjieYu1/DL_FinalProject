import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from Models.ESPCN import ESPCN

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(inChannels), int(outChannels), kernel_size=3)
        self.act = nn.ELU()

    def forward(self, x):
        if self.activation:
            out = self.act(self.conv(self.pad(x)))
        else:
            out = self.conv(self.pad(x))
        return out

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        espcn=False
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        if espcn:
            self.upsample = ESPCN(2, in_channels, in_channels, activation=False)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class DepthDecoderModelUNET(nn.Module):
    def __init__(self, numChannelsEncoder, espcn=False):
        super(DepthDecoderModelUNET, self).__init__()
        self.numChannelsEncoder = numChannelsEncoder
        self.espcn = espcn
        self.numChannelsDecoder = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for layer in range(4, -1, -1):
            inChannels = self.numChannelsEncoder[-1] if layer == 4 else self.numChannelsDecoder[layer+1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 0)] = ConvBlock(inChannels, outChannels, activation=True)
            inChannels = self.numChannelsDecoder[layer]
            if self.espcn:
                self.convs[("espcn", layer)] = ESPCN(2, outChannels, inChannels, activation=False)
            if layer > 0:
                inChannels += self.numChannelsEncoder[layer-1]
            outChannels = self.numChannelsDecoder[layer]
            self.convs[("upconv", layer, 1)] = ConvBlock(inChannels, outChannels, activation=True)
        for scale in range(4):
            self.convs[("dispconv", scale)] = ConvBlock(self.numChannelsDecoder[scale], 1, activation=False)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, inputFeatures):
        self.outputs = {}
        x = inputFeatures[-1]
        for layer in range(4, -1, -1):
            x = self.convs[("upconv", layer, 0)](x)
            if self.espcn:
                x = [self.convs[("espcn", layer)](x)]
            else:
                x = [nn.functional.interpolate(x, scale_factor=2, mode="nearest")]
            if layer > 0:
                x += [inputFeatures[layer-1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", layer, 1)](x)
            if layer < 4:
                out = self.convs[("dispconv", layer)](x)
                self.outputs[("disp", layer)] = torch.sigmoid(out)
        return self.outputs
    
class PoseDecoderModel(nn.Module):
    def __init__(self, numChannelsEncoder, numFeatures=1, numFrames=2):
        super(PoseDecoderModel, self).__init__()
        self.numChannelsEncoder = numChannelsEncoder
        self.numFeaturesInput = numFeatures
        self.numFramesPredict = numFrames
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.numChannelsEncoder[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(self.numFeaturesInput*256, 256, 3, 1, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6*self.numFramesPredict, 1)
        self.relu = nn.ReLU()
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, inputFeatures):
        lastFeatures = [feature[-1] for feature in inputFeatures]
        catFeatures = [self.relu(self.convs["squeeze"](feature)) for feature in lastFeatures]
        catFeatures = torch.cat(catFeatures, 1)
        out = catFeatures
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        bottleneck = out.unsqueeze(-1).unsqueeze(-1)
        out = 0.01 * out.view(-1, self.numFramesPredict, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation, bottleneck

class DepthDecoderModelUNETPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_channels,
        espcn=False
    ):
        super().__init__()
        n_blocks=5
        decoder_channels = [16, 32, 64, 128, 256]
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        use_batchnorm=True
        attention_type='scse'
        center=True

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, espcn=espcn)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        #blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
        #    self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        #)
        for scale in range(4):
#            blocks[f"dispconv_{scale}"] = ConvBlock(self.out_channels[4-scale], 1, activation=False)
            blocks[f"dispconv_{scale}"] = DecoderBlock(self.out_channels[4-scale-1], 0, 1, use_batchnorm=True, attention_type=None, espcn=False)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, features):

        #features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        #dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        outputs = {}
        for i in range(4):
            outputs[("disp", i)] = self.blocks[f"dispconv_{i}"](dense_x[f"x_0_{3-i}"])
        return outputs
