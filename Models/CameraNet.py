import torch
from torch import nn
import torchvision.transforms.functional as F
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), stride=2, padding=1)
        self.relu = nn.ReLU()
        self.out_channels = self.conv.out_channels

    def forward(self, in_tensor):
        return self.relu(self.conv(in_tensor))


class RefineMotionField(nn.Module):

    def __init__(self, motion_field_out_channels, layer_output_out_channels):
        super(RefineMotionField, self).__init__()
        self.conv1 = nn.Conv2d(motion_field_out_channels + layer_output_out_channels, max(4, layer_output_out_channels),
                               (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(motion_field_out_channels + layer_output_out_channels, max(4, layer_output_out_channels),
                               (3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, max(4, layer_output_out_channels), (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.conv1.out_channels + self.conv3.out_channels, motion_field_out_channels, (1, 1),
                               stride=1)

    def forward(self, motion_field, layer_output):
        h, w = layer_output.shape[2:]
        upsampled_motion_field = F.resize(motion_field, [h, w], interpolation=F.InterpolationMode.BILINEAR)
        conv_input = torch.cat([upsampled_motion_field, layer_output], dim=1)
        conv_output = self.conv1(conv_input)
        conv_input = self.conv2(conv_input)
        conv_output2 = self.conv3(conv_input)
        conv_output = torch.cat([conv_output, conv_output2], dim=1)
        conv_output = self.conv4(conv_output)
        return upsampled_motion_field + conv_output


class CameraIntrinsics(nn.Module):

    def __init__(self, bottleneck_channels=None, h=None, w=None):
        super(CameraIntrinsics, self).__init__()
        self.focal_lengths_conv = nn.Conv2d(bottleneck_channels, 2, (1, 1), stride=1)
        self.focal_lengths_act = nn.Softplus()
        self.offsets_conv = nn.Conv2d(bottleneck_channels, 2, (1, 1), stride=1, bias=False)
        self.scaling = torch.from_numpy(np.array([[w, h]])).float()
        self.last_2nd_row = torch.Tensor([[[0.0, 0.0, 1.0]]])
        self.last_row = torch.Tensor([[[0.0, 0.0, 0.0]]])
        self.last_col = torch.Tensor([[[0.0], [0.0], [0.0], [1.0]]])

    def forward(self, bottleneck):
        self.scaling = self.scaling.to(bottleneck.device)
        self.last_2nd_row = self.last_2nd_row.to(bottleneck.device)
        self.last_row = self.last_row.to(bottleneck.device)
        self.last_col = self.last_col.to(bottleneck.device)
        focal_lengths = self.focal_lengths_act(self.focal_lengths_conv(bottleneck)).squeeze(3).squeeze(2)
        focal_lengths *= self.scaling
        offsets = self.offsets_conv(bottleneck).squeeze(3).squeeze(2) + 0.5
        offsets *= self.scaling
        offsets = offsets.unsqueeze(-1)
        foci = torch.diag_embed(focal_lengths, dim1=-2, dim2=-1)
        intrinsic_mat = torch.cat([foci, offsets], dim=2)
        last_2nd_row = torch.tile(self.last_2nd_row, [bottleneck.shape[0], 1, 1])
        last_row = torch.tile(self.last_row, [bottleneck.shape[0], 1, 1])
        last_col = torch.tile(self.last_col, [bottleneck.shape[0], 1, 1])
        intrinsic_mat = torch.cat([intrinsic_mat, last_2nd_row, last_row], dim=1)
        intrinsic_mat = torch.cat([intrinsic_mat, last_col], dim=2)
        return intrinsic_mat


class CameraNet(nn.Module):

    def __init__(self, in_channels, h=192, w=640, refine=False):
        super(CameraNet, self).__init__()
        self.refine = refine
        self.rotation_scale, self.translation_scale = 0.01, 0.01
        self.conv1 = ConvBlock(2 * in_channels, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.conv4 = ConvBlock(64, 128)
        self.conv5 = ConvBlock(128, 256)
        self.backgroundMotion = nn.Conv2d(256, 6, (1, 1), stride=1, bias=False)
        self.cameraIntrinsics = CameraIntrinsics(self.conv5.out_channels, h, w)
        if self.refine:
            self.refineMotionField1 = RefineMotionField(3, self.conv5.out_channels)
            self.refineMotionField2 = RefineMotionField(3, self.conv4.out_channels)
            self.refineMotionField3 = RefineMotionField(3, self.conv3.out_channels)
            self.refineMotionField4 = RefineMotionField(3, self.conv2.out_channels)
            self.refineMotionField5 = RefineMotionField(3, self.conv1.out_channels)
            self.refineMotionField6 = RefineMotionField(3, 2 * in_channels)

    def forward(self, in_tensor):
        """
        Args: Input of shape [B, C, H, W] where C = 2*in_channels concatenated images
        Returns:
            Rotation of shape [B, 3]
            Translation of shape [B, 3, 1, 1]
            Residual Translation of shape [B, 3, H, W]
            Camera Intrinsics (K) of shape [B, ]
        """
        out1 = self.conv1(in_tensor)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        bottleneck = torch.mean(out5, dim=(2, 3), keepdim=True)
        background_motion = self.backgroundMotion(bottleneck)
        rotation = self.rotation_scale * background_motion[:, :3, 0, 0]
        translation = self.translation_scale * background_motion[:, 3:, :, :]
        intrinsics = self.cameraIntrinsics(bottleneck)
        if self.refine:
            residual_translation = self.refineMotionField1(translation, out5)
            residual_translation = self.refineMotionField2(residual_translation, out4)
            residual_translation = self.refineMotionField3(residual_translation, out3)
            residual_translation = self.refineMotionField4(residual_translation, out2)
            residual_translation = self.refineMotionField5(residual_translation, out1)
            residual_translation = self.refineMotionField6(residual_translation, in_tensor)
        else:
            residual_translation = None
        return rotation.unsqueeze(1).unsqueeze(1), translation.permute(0, 2, 3, 1), residual_translation, intrinsics


if __name__ == "__main__":
    c = CameraNet(3)
    z = torch.rand(7, 6, 192, 640)
    rot, trans, res_trans, K = c(z)
    print(rot.shape)
    print(trans.shape)
    print(K.shape)
    print(K)
