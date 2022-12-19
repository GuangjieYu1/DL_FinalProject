import numpy as np

import torch
import torch.nn as nn

class BackprojectDepth(nn.Module):
    def __init__(self, batchSize, height, width):
        super(BackprojectDepth, self).__init__()
        self.batchSize = batchSize
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.idCoordinates = np.stack(meshgrid, axis=0).astype(np.float32)
        self.idCoordinates = nn.Parameter(torch.from_numpy(self.idCoordinates), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batchSize, 1, self.height*self.width),
                                 requires_grad=False)
        self.pixelCoordinates = torch.unsqueeze(torch.stack([self.idCoordinates[0].view(-1),
                                                             self.idCoordinates[1].view(-1)], 0), 0)
        self.pixelCoordinates = self.pixelCoordinates.repeat(self.batchSize, 1, 1)
        self.pixelCoordinates = nn.Parameter(torch.cat([self.pixelCoordinates, self.ones], 1),
                                             requires_grad=False)

    def forward(self, depth, inv_K):
        cameraPoints = torch.matmul(inv_K[:, :3, :3], self.pixelCoordinates)
        cameraPoints = depth.view(self.batchSize, 1, -1)*cameraPoints
        cameraPoints = torch.cat([cameraPoints, self.ones], 1)
        return cameraPoints
