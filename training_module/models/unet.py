import torch
from torch import nn
from collections import OrderedDict
from ._utils import SpatialCoordinatesExpectation


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, init_features=64, levels=5, dropout=0.25, avg_polling=False):
        super(UNet, self).__init__()

        features = init_features

        enconders = []
        pools = []
        upconvs = []
        decoders = []

        for layer in range(levels):
            if layer == 0:
                enconders.append(
                    UNet._block(in_channels, features, name="enc" + str(layer + 1))
                )
            else:
                enconders.append(
                    UNet._block(features * (2 ** (layer - 1)), features * (2 ** layer), name="enc" + str(layer + 1))
                )
            if avg_polling:
                pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for layer in range(levels, 0, -1):
            upconvs.append(
                nn.ConvTranspose2d(features * (2 ** layer), features * (2 ** (layer - 1)), kernel_size=2, stride=2)
            )
            decoders.append(
                UNet._block(features * (2 ** layer), features * (2 ** (layer - 1)), name="dec" + str(layer + 1))
            )

        self.enconders = nn.Sequential(*enconders)
        self.pools = nn.Sequential(*pools)

        self.bottleneck = UNet._block(features * (2 ** (levels - 1)), features * (2 ** levels), name="bottleneck")

        self.upconvs = nn.Sequential(*upconvs)
        self.decoders = nn.Sequential(*decoders)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

        self.spatial_expectation = SpatialCoordinatesExpectation()

    def forward(self, x):

        encoder_output = []

        for level, encoder in enumerate(self.enconders):
            if level == 0:
                encoder_output.append(encoder(x))
            else:
                encoder_output.append(encoder(self.pools[level](encoder_output[level - 1])))
            if level >= len(self.enconders) - 2:
                encoder_output[level] = self.dropout(encoder_output[level])

        bottleneck = self.bottleneck(self.pools[-1](encoder_output[-1]))

        for level in range(len(self.decoders)):
            if level == 0:
                output = self.upconvs[level](bottleneck)
            else:
                output = self.upconvs[level](output)

            output = torch.cat((output, encoder_output[len(encoder_output) - level - 1]), dim=1)

            output = self.decoders[level](output)

        heatmaps = self.conv(output)

        heatmaps, coordinates = self.spatial_expectation(heatmaps)

        return heatmaps, coordinates

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )