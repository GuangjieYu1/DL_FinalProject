import torch
import torch.nn as nn

from Losses.ReprojectionLoss import ReprojectionLoss
from Losses.SmoothLoss import SmoothLoss

class Loss(nn.Module):
    def __init__(self, numScales, frameIdxs, device, automasking=False):
        super(Loss, self).__init__()
        self.numScales = numScales
        self.frameIdxs = frameIdxs
        self.device = device
        self.automasking = automasking
        self.reprojectionLoss = ReprojectionLoss()
        self.smoothLoss = SmoothLoss()

    def forward(self, inputs, outputs):
        losses = {}
        totalLoss = 0
        sourceScale = 0
        for scale in range(self.numScales):
            loss = 0
            reprojectionLoss = []
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, sourceScale)]
            for frameIdx in self.frameIdxs[1:]:
                pred = outputs[("color", frameIdx, scale)]
                reprojectionLoss.append(self.reprojectionLoss(pred, target))
            reprojectionLoss = torch.cat(reprojectionLoss, 1)
            reprojectionLoss = reprojectionLoss.mean(1, keepdim=True)
            if self.automasking:
                identityReprojectionLoss = []
                for frameIdx in self.frameIdxs[1:]:
                    pred = inputs[("color", frameIdx, sourceScale)]
                    identityReprojectionLoss.append(self.reprojectionLoss(pred, target))
                identityReprojectionLoss = torch.cat(identityReprojectionLoss, 1)
                identityReprojectionLoss = identityReprojectionLoss.mean(1, keepdim=True)
                identityReprojectionLoss += torch.randn(identityReprojectionLoss.shape, device=self.device) * 0.00001
                combined = torch.cat((identityReprojectionLoss, reprojectionLoss), 1)
            else:
                combined = reprojectionLoss
            if combined.shape[1] == 1:
                toOptimise = combined
            else:
                toOptimise, idxs = torch.min(combined, dim=1)
            if self.automasking:
                outputs["identity_selection/{}".format(scale)] = (idxs > identityReprojectionLoss.shape[1] - 1).float()
            loss += toOptimise.mean()
            meanDisp = disp.mean(2, True).mean(3, True)
            normDisp = disp / (meanDisp + 1e-7)
            smoothLoss = self.smoothLoss(normDisp, color)
            loss += (1e-3 * smoothLoss)/(2**scale)
            totalLoss += loss
            losses["loss/{}".format(scale)] = loss
        totalLoss /= self.numScales
        losses["loss"] = totalLoss
        return losses
