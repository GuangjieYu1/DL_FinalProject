import numpy as np

import torch
import torch.nn as nn

class Project3D(nn.Module):
    def __init__(self, batchSize, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batchSize = batchSize
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cameraPoints = torch.matmul(P, points)
        pixelCoordinates = cameraPoints[:, :2, :]/(cameraPoints[:, 2, :].unsqueeze(1) + self.eps)
        pixelCoordinates = pixelCoordinates.view(self.batchSize, 2, self.height, self.width)
        pixelCoordinates = pixelCoordinates.permute(0, 2, 3, 1)
        pixelCoordinates[..., 0] /= self.width - 1
        pixelCoordinates[..., 1] /= self.height - 1
        pixelCoordinates = (pixelCoordinates - 0.5)*2
        return pixelCoordinates
