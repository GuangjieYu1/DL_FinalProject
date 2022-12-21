import torch
import torch.nn as nn

from Losses.SSIM import SSIM

class ReprojectionLoss(nn.Module):
    def __init__(self):
        super(ReprojectionLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, prediction, target):
        absDiff = torch.abs(target - prediction)
        l1Loss = absDiff.mean(1, True)
        ssim_loss = self.ssim(prediction, target).mean(1, True)
        return 0.85*ssim_loss + 0.15*l1Loss
