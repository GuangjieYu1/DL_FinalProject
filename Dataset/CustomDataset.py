import random
random.seed(7)
import PIL.Image as pil

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import albumentations as A

from utils import pilLoader

class CustomDataset(data.Dataset):
    def __init__(self, dataPath, filenames, height, width, frameIdxs, numScales, train=False, weather_aug=False):
        super(CustomDataset, self).__init__()
        self.dataPath = dataPath
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frameIdxs = frameIdxs
        self.numScales = numScales
        self.train = train
        self.weather_aug = weather_aug
        self.interpolation = T.InterpolationMode.LANCZOS
        self.loader = pilLoader
        self.toTensor = T.ToTensor()
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.resize = {}
        for scaleNum in range(self.numScales):
            scale = 2**scaleNum
            self.resize[scaleNum] = T.Resize((self.height//scale, self.width//scale),
                                             interpolation=self.interpolation)
        self.transforms =   {
                            "ColorJitter": T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue),
                            "RandomRain": A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                            "RandomSnow": A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),
                            "RandomFog": A.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.6, alpha_coef=0.1, p=1),
                            "RandomSunFlare": A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, src_radius=200, p=1)
                            }
        self.loadDepth = self.checkDepth()

    def preprocess(self, inputs, colorAugmentations, extraAugs):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.numScales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.toTensor(frame)
                frame = colorAugmentations(frame)
                if extraAugs:
                    extraAugs = A.Compose(extraAugs)
                    frame = extraAugs(image=np.array(frame))
                    frame = pil.fromarray(frame['image'])
                inputs[(n + "_aug", im, i)] = self.toTensor(frame)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        colorAugmentationsFlag = self.train and random.random() > 0.5
        flipFlag = self.train and random.random() > 0.5
        rainFlag = self.train and random.random() > 0.7
        snowFlag = self.train and (not rainFlag) and random.random() > 0.7
        fogFlag = self.train and random.random() > 0.7
        sunFlareFlag = self.train and random.random() > 0.7
        line = self.filenames[index].split()
        directory = line[0]
        frameIdx = 0
        side = None
        if len(line) == 3:
            frameIdx = int(line[1])
            side = line[2]
        for fi in self.frameIdxs:
            if fi == "s":
                otherSide = {"r": "l", "l": "r"}[side]
                inputs[("color", fi, -1)] = self.getColor(directory, frameIdx, otherSide, flipFlag)
            else:
                inputs[("color", fi, -1)] = self.getColor(directory, frameIdx + fi, side, flipFlag)
        for scale in range(self.numScales):
            K = self.K.copy()
            K[0, :] *= self.width//(2**scale)
            K[1, :] *= self.height//(2**scale)
            inverseK = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inverseK)
        colorAugmentations = (lambda x: x)
        if colorAugmentationsFlag:
            colorAugmentations = self.transforms["ColorJitter"]
        extraAugs = []
        if self.weather_aug:
            if rainFlag:
                extraAugs.append(self.transforms["RandomRain"])
            if snowFlag:
                extraAugs.append(self.transforms["RandomSnow"])
            if fogFlag:
                extraAugs.append(self.transforms["RandomFog"])
            if sunFlareFlag:
                extraAugs.append(self.transforms["RandomSunFlare"])
        self.preprocess(inputs, colorAugmentations, extraAugs)
        for fi in self.frameIdxs:
            del inputs[("color", fi, -1)]
            del inputs[("color_aug", fi, -1)]
        if self.loadDepth:
            depthGroundTruth = self.getDepth(directory, frameIdx, side, flipFlag)
            inputs["depth_gt"] = np.expand_dims(depthGroundTruth, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        if "s" in self.frameIdxs:
            stereoT = np.eye(4, dtype=np.float32)
            baselineSign = -1 if flipFlag else 1
            sideSign = -1 if side == "l" else 1
            stereoT[0, 3] = sideSign * baselineSign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereoT)
        return inputs

    def checkDepth(self):
        raise NotImplementedError

    def getColor(self, directory, frameIdx, otherSide, flip):
        raise NotImplementedError

    def getDepth(self, directory, frameIdx, otherSide, flip):
        raise NotImplementedError
