import os
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3, Inception3
from ._utils import SpatialCoordinatesExpectation, device


'''https://github.com/rosinality/stylegan2-pytorch/blob/master/calc_inception.py'''


class Inception3_OutputChanged(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # N x 1 x 299 x 299
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2b_3x3(x)  # N x 32 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 64 x 147 x 147

        x = self.Conv2d_3b_1x1(x)  # N x 64 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 80 x 73 x 73
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # N x 192 x 35 x 35

        x = self.Mixed_5b(x)  # N x 192 x 35 x 35
        x = self.Mixed_5c(x)  # N x 256 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35

        x = self.Mixed_6a(x)  # N x 288 x 35 x 35
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17

        x = self.Mixed_7a(x)  # N x 768 x 17 x 17
        x = self.Mixed_7b(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8

        return x


class HeatmapOutput(nn.Module):
    def __init__(self, num_landmarks):
        super(HeatmapOutput, self).__init__()
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv1 = nn.Conv2d(2048, num_landmarks, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False, device=device)

    def forward(self, x):
        x = self.upscale(x)
        x = self.conv1(x)
        return x


class CustomInceptionV3(nn.Module):
    def __init__(self):
        super(CustomInceptionV3, self).__init__()

        self.inception = inception_v3(torchvision.models.Inception_V3_Weights.DEFAULT)

        self.inception_heatmaps = Inception3_OutputChanged()
        self.inception_heatmaps.load_state_dict(self.inception.state_dict())

        for param in self.inception_heatmaps.parameters():
            param.requires_grad = False

        self.inception_heatmaps.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            
        self.heatmap_output = HeatmapOutput(num_landmarks=int(os.getenv('NUM_KPTS')))
        self.spatial_expectation = SpatialCoordinatesExpectation()

    def forward(self, x):
        x = self.inception_heatmaps(x)
        x = self.heatmap_output(x)
        x, coordinates = self.spatial_expectation(x)
        return x, coordinates


def get_inception_v3_heatmap():
    inception_model = CustomInceptionV3()
    return inception_model
