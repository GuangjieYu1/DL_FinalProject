import os
import numpy as np
import skimage.transform as T
import PIL.Image as pil

from Dataset.CustomDataset import CustomDataset
from utils import generateDepthMap

class KITTI(CustomDataset):
    def __init__(self, *args, **kwargs):
        super(KITTI, self).__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0.00, 0.50, 0.00],
                           [0.00, 1.92, 0.50, 0.00],
                           [0.00, 0.00, 1.00, 0.00],
                           [0.00, 0.00, 0.00, 1.00]], dtype=np.float32)
        self.fullResolution = (1242, 375)
        self.sideMap = {"2": 2, "3": 3, "l": 2, "r": 3}

    def getImagePath(self, directory, frameIdx, side):
        baseFileName = "{:010d}.png".format(frameIdx)
        sidename = "image_0{}/data".format(self.sideMap[side])
        imagePath = os.path.join(self.dataPath, directory, sidename, baseFileName)
        return imagePath

    def checkDepth(self):
        line = self.filenames[0].split()
        sceneName = line[0]
        frameIdx = int(line[1])
        velodyneFileName = "velodyne_points/data/{:010d}.bin".format(int(frameIdx))
        velodyneFilePath = os.path.join(self.dataPath, sceneName, velodyneFileName)
        return os.path.isfile(velodyneFilePath)

    def getDepth(self, directory, frameIdx, side, flip):
        calibrationPath = os.path.join(self.dataPath, directory.split("/")[0])
        velodyneFileName = "velodyne_points/data/{:010d}.bin".format(int(frameIdx))
        velodyneFilePath = os.path.join(self.dataPath, directory, velodyneFileName)
        depthGroundTruth = generateDepthMap(calibrationPath, velodyneFilePath, self.sideMap[side])
        depthGroundTruth = T.resize(depthGroundTruth, self.fullResolution[::-1], order=0,
                                    mode="constant", preserve_range=True)
        if flip:
            depthGroundTruth = np.fliplr(depthGroundTruth)
        return depthGroundTruth

    def getColor(self, directory, frameIdx, side, flip):
        imagePath = self.getImagePath(directory, frameIdx, side)
        color = self.loader(imagePath)
        if flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
