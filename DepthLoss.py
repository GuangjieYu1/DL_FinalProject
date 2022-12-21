import numpy as np
import torch
import torch.nn as nn

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.depthMetricNames = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    def computeDepthErrors(self, depthGroundTruth, depthPred):
        threshold = torch.max((depthGroundTruth/depthPred), (depthPred/depthGroundTruth))
        a1 = (threshold < 1.25).float().mean()
        a2 = (threshold < 1.25**2).float().mean()
        a3 = (threshold < 1.25**3).float().mean()
        rootMeanSquaredError = (depthGroundTruth - depthPred)**2
        rootMeanSquaredError = torch.sqrt(rootMeanSquaredError.mean())
        rootMeanSquaredErrorLog = (torch.log(depthGroundTruth) - torch.log(depthPred))**2
        rootMeanSquaredErrorLog = torch.sqrt(rootMeanSquaredErrorLog.mean())
        absolute = torch.mean(torch.abs(depthGroundTruth - depthPred)/depthGroundTruth)
        squared = torch.mean(((depthGroundTruth - depthPred)**2)/depthGroundTruth)
        return absolute, squared, rootMeanSquaredError, rootMeanSquaredErrorLog, a1, a2, a3

    def forward(self, inputs, outputs):
        depthPred = outputs[("depth", 0, 0)]
        depthPred = torch.clamp(nn.functional.interpolate(depthPred, [375, 1242], mode='bilinear', align_corners=False), 1e-3, 80)
        depthPred = depthPred.detach()
        depthGroundTruth = inputs["depth_gt"]
        mask = depthGroundTruth > 0
        cropMask = torch.zeros_like(mask)
        cropMask[:, :, 153:371, 44:1197] = 1
        mask = mask * cropMask
        depthGroundTruth = depthGroundTruth[mask]
        depthPred = depthPred[mask]
        depthPred *= torch.median(depthGroundTruth)/torch.median(depthPred)
        depthPred = torch.clamp(depthPred, 1e-3, 80)
        depthErrors = self.computeDepthErrors(depthGroundTruth, depthPred)
        losses = {}
        for i, name in enumerate(self.depthMetricNames):
            losses[name] = np.array(depthErrors[i].cpu())
        return losses
