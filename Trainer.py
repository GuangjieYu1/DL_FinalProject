import os
import sys
import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from Models.EncoderModel import EncoderModelResNet, EncoderModelConvNeXt
from Models.DecoderModel import DepthDecoderModelUNET, DepthDecoderModelUNETPlusPlus, PoseDecoderModel
from Models.CameraNet import CameraNet, CameraIntrinsics
from Models.BackprojectDepth import BackprojectDepth
from Models.Project3D import Project3D
from Dataset.KITTI import KITTI
from Losses.Loss import Loss
from Losses.DepthLoss import DepthLoss
from utils import secondsToHM, transformParameters, dispToDepth, normalizeImage
from config import Config

class Trainer:
    def __init__(self,config_path):
        self.configure(config_path)
        self.createModels()
        self.setupProjections()
        self.setupLosses()
        self.createOptimizer()
        self.loadDataset()
        self.setupLogging()
        
    def configure(self, config_path):
        self.config = Config(config_path)()
        self.modelName = self.config["modelName"]
        print("Starting up Model : {}".format(self.modelName))
        self.height = 192
        self.width = 640
        self.frameIdxs = [0, -1, 1]
        self.numScales = len([0, 1, 2, 3])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on device: {}".format(self.device))
        
    def createModels(self):
        totalTrainableParams = 0
        self.trainableParameters = []
        self.models = {}
        self.models["encoder"] = eval(self.config["Model"]["Encoder"])()
        depthDecoder = self.config["Model"]["DepthDecoder"] + self.config["Model"]["Arch"]
        self.models["decoder"] = eval(depthDecoder)(self.models["encoder"].numChannels, espcn=self.config["Model"]["ESPCN"])
        if self.config["Model"]["PoseDecoder"] == "CameraNet":
            if self.config["CameraNet"]["InputType"] == "EncoderLast":
                self.models["pose"] = CameraNet(self.models["encoder"].numChannels[-1], h=self.height//(2**(self.numScales+1)), w=self.width//(2**(self.numScales+1)))
            elif self.config["CameraNet"]["InputType"] == "Images":
                self.models["pose"] = CameraNet(3, h=self.height, w=self.width)
        else:
            self.models["pose"] = eval(self.config["Model"]["PoseDecoder"])(self.models["encoder"].numChannels, 2, 1)
        additional = self.config["Model"]["Additional"]
        if additional:
            params = ""
            for key, value in self.config[additional].items():
                params += str(key) + "=" + str(value) + ","
            params = params[:-1]
            self.models[additional] = eval(additional + "(" + params + ")")
        for key, model in self.models.items():
            self.models[key] = self.models[key].to(self.device)
            self.trainableParameters += list(model.parameters())
            totalTrainableParams += sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Trainable Parameters: {}".format(totalTrainableParams))
        
    def setupLosses(self):
        self.losses = {}
        self.losses["Loss"] = Loss(self.numScales, self.frameIdxs, self.device, self.config["Model"]["Automasking"])
        self.losses["Depth"] = DepthLoss()
        for key, model in self.losses.items():
            self.losses[key] = self.losses[key].to(self.device)
        
    def createOptimizer(self):
        self.optimizer = eval("optim." + self.config["Optimizer"]["Type"])(self.trainableParameters, \
                                                                           lr=float(self.config["Optimizer"]["LR"]), \
                                                                           weight_decay=float(self.config["Optimizer"]["WeightDecay"]))
        self.lrScheduler = eval("optim.lr_scheduler." + self.config["Scheduler"]["Type"])(self.optimizer, self.config["Scheduler"]["NumEpochs"])
        
    def setupProjections(self):
        self.backprojectDepth = {}
        self.project3d = {}
        for scale in range(self.numScales):
            h = self.height // (2**scale)
            w = self.width // (2**scale)
            self.backprojectDepth[scale] = BackprojectDepth(int(self.config["DataLoader"]["BatchSize"]), h, w)
            self.backprojectDepth[scale] = self.backprojectDepth[scale].to(self.device)
            self.project3d[scale] = Project3D(int(self.config["DataLoader"]["BatchSize"]), h, w)
            self.project3d[scale] = self.project3d[scale].to(self.device)

    def setupLogging(self):
        self.writers = {}
        logPath = self.config["Logger"]["Path"]
        for mode in ["train", "val"]:
            path = os.path.join(logPath, self.modelName, mode)
            if os.path.exists(path):
                shutil.rmtree(path)
            self.writers[mode] = SummaryWriter(path)
        
    def readlines(self, path):
        with open(path, "r") as f:
            lines = f.read().splitlines()
        return lines

    def loadDataset(self):
        self.dataset = KITTI
        dataPath = self.config["DataLoader"]["Path"]
        filepath = os.path.join(dataPath, "splits", "eigen_zhou", "{}_files.txt")
        trainFilenames = self.readlines(filepath.format("train"))
        valFilenames = self.readlines(filepath.format("val"))
        numTrain = len(trainFilenames)
        self.numSteps = (numTrain//int(self.config["DataLoader"]["BatchSize"]))*int(self.config["Trainer"]["Epochs"])
        weather_aug = self.config["DataLoader"].get("WeatherAug", False)
        trainDataset = self.dataset(dataPath, trainFilenames, self.height, self.width,
                                    self.frameIdxs, 4, train=True, weather_aug=weather_aug)
        valDataset = self.dataset(dataPath, valFilenames, self.height, self.width, self.frameIdxs,
                                  4, train=False, weather_aug=False)
        self.trainLoader = DataLoader(trainDataset, int(self.config["DataLoader"]["BatchSize"]), shuffle=True, num_workers=14, pin_memory=True, drop_last=True)
        self.valLoader = DataLoader(valDataset, int(self.config["DataLoader"]["BatchSize"]), shuffle=True, num_workers=14, pin_memory=True, drop_last=True)
        self.valIterator = iter(self.valLoader)
        print("Total Number of Steps to Run: {}".format(self.numSteps))

    def setTrain(self):
        for model in self.models.values():
            model.train()

    def setEval(self):
        for model in self.models.values():
            model.eval()
            
    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for lossname, value in losses.items():
            writer.add_scalar("{}".format(lossname), value, self.step)
        for frameIdx in self.frameIdxs:
            writer.add_image("color_{}".format(frameIdx), inputs[("color", frameIdx, 0)][0].data, self.step)
            if frameIdx != 0:
                writer.add_image("color_pred_{}".format(frameIdx), outputs[("color", frameIdx, 0)][0].data, self.step)
        writer.add_image("disp", normalizeImage(outputs[("disp", 0)][0]), self.step)
        if self.config["Model"]["Automasking"]:
            writer.add_image("automask", outputs[("identity_selection/0")][0][None, ...], self.step)
            
    
    def logTime(self, batchIdx, duration, loss):
        samplesPerSec = int(self.config["DataLoader"]["BatchSize"]) / duration
        totalTime = time.time() - self.startTime
        timeLeft = (self.numSteps / self.step - 1.0)*totalTime if self.step > 0 else 0
        logString = "Epoch : {:>3} | Batch : {:>7} | Step : {:>10} | examples/s: {:5.1f} | loss : {:.5f} | time elapsed: {} | time left: {}"
        print(logString.format(self.epoch, batchIdx, self.step, samplesPerSec, loss, secondsToHM(totalTime), secondsToHM(timeLeft)))

    def saveModel(self):
        outpath = os.path.join(self.config["Trainer"]["ModelSavePath"], self.modelName, "weights_{}".format(self.epoch))
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            shutil.rmtree(outpath)
        for name, model in self.models.items():
            savePath = os.path.join(outpath, "{}.pth".format(name))
            toSave = model.state_dict()
            if name == "encoder":
                toSave["height"] = self.height
                toSave["width"] = self.width
            torch.save(toSave, savePath)
        savePath = os.path.join(outpath, "adam.pth")
        torch.save(self.optimizer.state_dict(), savePath)
        
    def predictPoses(self, inputs, features):
        outputs = {}
        poseFeatures = {fi: features[fi] for fi in self.frameIdxs}
        for fi in self.frameIdxs[1:]:
            if fi < 0:
                poseInputs = [poseFeatures[fi], poseFeatures[0]]
            else:
                poseInputs = [poseFeatures[0], poseFeatures[fi]]
            if self.config["Model"]["PoseDecoder"] == "CameraNet":
                if self.config["CameraNet"]["InputType"] == "EncoderLast":
                    poseInputs = torch.cat([f[-1] for f in poseInputs], dim=1)
                    axisangle, translation, _, intrinsics = self.models["pose"](poseInputs)
                    intrinsics[:, 0] *= 2**(self.numScales - 1)
                    intrinsics[:, 1] *= 2**(self.numScales - 1)
                elif self.config["CameraNet"]["InputType"] == "Images":
                    poseInputs = [inputs[("color_aug", 0, 0)], inputs[("color_aug", fi, 0)]]
                    if fi < 0:
                        poseInputs = poseInputs[::-1]
                    poseInputs = torch.cat(poseInputs, dim=1)
                    axisangle, translation, _, intrinsics = self.models["pose"](poseInputs)
                outputs[("K", fi, 0)] = intrinsics
                outputs[("inv_K", fi, 0)] = torch.linalg.pinv(intrinsics)
            else:
                axisangle, translation, bottleneck = self.models["pose"](poseInputs)
                if self.config["Model"]["Additional"] == "CameraIntrinsics":
                    intrinsics = self.models["CameraIntrinsics"](bottleneck)
                    intrinsics[:, 0] *= 2**(self.numScales - 1)
                    intrinsics[:, 1] *= 2**(self.numScales - 1)
                    outputs[("K", fi, 0)] = intrinsics
                    outputs[("inv_K", fi, 0)] = torch.linalg.pinv(intrinsics)
            outputs[("axisangle", 0, fi)] = axisangle
            outputs[("translation", 0, fi)] = translation
            outputs[("cam_T_cam", 0, fi)] = transformParameters(axisangle[:, 0], translation[:, 0], invert=(fi<0))
            
        return outputs

    def generateImagePredictions(self, inputs, outputs):
        for scale in range(self.numScales):
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear",
                                 align_corners=False)
            sourceScale = 0
            _, depth = dispToDepth(disp, 0.1, 100.0)
            outputs[("depth", 0, scale)] = depth
            for i, frameIdx in enumerate(self.frameIdxs[1:]):
                T = outputs[("cam_T_cam", 0, frameIdx)]
                if self.config["Model"]["K_Trainable"] == True:
                    cameraPoints = self.backprojectDepth[sourceScale](depth, outputs[("inv_K", frameIdx, sourceScale)])
                    pixelCoordinates = self.project3d[sourceScale](cameraPoints, outputs[("K", frameIdx, sourceScale)], T)
                else:
                    cameraPoints = self.backprojectDepth[sourceScale](depth, inputs[("inv_K", sourceScale)])
                    pixelCoordinates = self.project3d[sourceScale](cameraPoints, inputs[("K", sourceScale)], T)
                outputs[("sample", frameIdx, scale)] = pixelCoordinates
                outputs[("color", frameIdx, scale)] = F.grid_sample(inputs[("color", frameIdx, sourceScale)],
                                                                    outputs[(("sample", frameIdx, scale))],
                                                                    padding_mode="border", align_corners=False)
                if self.config["Model"]["Automasking"]:
                    outputs[("color_identity", frameIdx, scale)] = inputs[("color", frameIdx, sourceScale)]

    def processBatch(self, inputs):
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        origScaleColorAug = torch.cat([inputs[("color_aug", fi, 0)] for fi in self.frameIdxs])
        allFrameFeatures = self.models["encoder"](origScaleColorAug)
        allFrameFeatures = [torch.split(f, int(self.config["DataLoader"]["BatchSize"])) for f in allFrameFeatures]
        features = {}
        for i, frameIdx in enumerate(self.frameIdxs):
            features[frameIdx] = [f[i] for f in allFrameFeatures]
        outputs = self.models["decoder"](features[0])
        outputs.update(self.predictPoses(inputs, features))
        self.generateImagePredictions(inputs, outputs)
        losses = self.losses["Loss"](inputs, outputs)
        return outputs, losses

    def runEpoch(self):
        self.setTrain()
        for batchIdx, inputs in enumerate(self.trainLoader):
            startTime = time.time()
            outputs, losses = self.processBatch(inputs)
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            duration = time.time() - startTime
            early_phase = batchIdx % 200 == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0
            if early_phase or late_phase:
                self.logTime(batchIdx, duration, losses["loss"].cpu().data)
                losses.update(self.losses["Depth"](inputs, outputs))
                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        self.lrScheduler.step()

    def train(self):
        print("Total Steps : {}".format(self.numSteps))
        self.epoch = 0
        self.step = 0
        self.startTime = time.time()
        for self.epoch in range(int(self.config["Trainer"]["Epochs"])):
            print("Training --- Epoch : {}".format(self.epoch))
            self.runEpoch()
            self.saveModel()

    def val(self):
        self.setEval()
        try:
            inputs = self.valIterator.next()
        except:
            self.valIterator = iter(self.valLoader)
            inputs = self.valIterator.next()
        with torch.no_grad():
            outputs, losses = self.processBatch(inputs)
            losses.update(self.losses["Depth"](inputs, outputs))
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
        self.setTrain()

if __name__ == "__main__":
    Trainer(sys.argv[1]).train()