import torch
from torch import nn
from kornia.geometry.subpix import spatial_softmax2d
from kornia.geometry.subpix import spatial_expectation2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SpatialCoordinatesExpectation(nn.Module):
    def __init__(self):
        super(SpatialCoordinatesExpectation, self).__init__()
        self.spatial_softmax = spatial_softmax2d
        self.spatial_expectation = spatial_expectation2d

    def forward(self, x):
        x = self.spatial_softmax(x)
        coordinates = self.spatial_expectation(x)
        return x, coordinates