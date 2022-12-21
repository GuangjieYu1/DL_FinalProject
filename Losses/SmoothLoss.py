import torch
import torch.nn as nn

class SmoothLoss(nn.Module):
    """Layer to compute the gradients of image pairs
    """
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, disparity, img):
        gradientDispX = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        gradientDispY = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])
        gradientImgX = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        gradientImgY = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        gradientDispX *= torch.exp(-gradientImgX)
        gradientDispY *= torch.exp(-gradientImgY)
        return gradientDispX.mean() + gradientDispY.mean()
