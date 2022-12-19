import os
import torch
import numpy as np
import PIL.Image as pil
from collections import Counter

def pilLoader(imagePath):
    with open(imagePath, 'rb') as f:
        image = pil.open(f)
        image = image.convert('RGB')
        return image
    
def readCalibrationFile(calibrationFilePath):
    floatChars = set("0123456789.e+- ")
    data = {}
    with open(calibrationFilePath, "r") as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if floatChars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def loadVelodyneFile(velodyneFilePath):
	points = np.fromfile(velodyneFilePath, dtype=np.float32).reshape(-1, 4)
	points[:, 3] = 1.0
	return points

def subscriptsToIndices(matrixSize, rowSubscripts, colSubscripts):
	m, n = matrixSize
	return rowSubscripts * (n-1) + colSubscripts - 1

def generateDepthMap(calibrationDirectory, velodyneFilePath, cam=2, velodyneDepth=False):
    cam2cam = readCalibrationFile(os.path.join(calibrationDirectory, 'calib_cam_to_cam.txt'))
    velo2cam = readCalibrationFile(os.path.join(calibrationDirectory, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    imgShape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    velodyne = loadVelodyneFile(velodyneFilePath)
    velodyne = velodyne[velodyne[:, 0] >= 0, :]
    velodynePointsImg = np.dot(P_velo2im, velodyne.T).T
    velodynePointsImg[:, :2] = velodynePointsImg[:, :2]/velodynePointsImg[:, 2][..., np.newaxis]
    if velodyneDepth:
        velodynePointsImg[:, 2] = velodyne[:, 0]
    velodynePointsImg[:, 0] = np.round(velodynePointsImg[:, 0]) - 1
    velodynePointsImg[:, 1] = np.round(velodynePointsImg[:, 1]) - 1
    velodyneIndices = (velodynePointsImg[:, 0] >= 0) & (velodynePointsImg[:, 1] >= 0)
    velodyneIndices = velodyneIndices & (velodynePointsImg[:, 0] < imgShape[1]) & (velodynePointsImg[:, 1] < imgShape[0])
    velodynePointsImg = velodynePointsImg[velodyneIndices, :]
    depth = np.zeros((imgShape[:2]))
    depth[velodynePointsImg[:, 1].astype(np.int), velodynePointsImg[:, 0].astype(np.int)] = velodynePointsImg[:, 2]
    indices = subscriptsToIndices(depth.shape, velodynePointsImg[:, 1], velodynePointsImg[:, 0])
    duplicateIndices = [item for item, count in Counter(indices).items() if count > 1]
    for di in duplicateIndices:
        points = np.where(indices == di)[0]
        x_loc = int(velodynePointsImg[points[0], 0])
        y_loc = int(velodynePointsImg[points[0], 1])
        depth[y_loc, x_loc] = velodynePointsImg[points, 2].min()
    depth[depth < 0] = 0
    return depth

def secondsToHM(durationSecs):
    t = int(durationSecs)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

def rotationFromAxisAngle(axisangle):
    angle = torch.norm(axisangle, 2, 2, True)
    axis = axisangle / (angle + 1e-7)
    cosAngle = torch.cos(angle)
    sinAngle = torch.sin(angle)
    complementCos = 1 - cosAngle
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    xs = x * sinAngle
    ys = y * sinAngle
    zs = z * sinAngle
    xcomplementCos = x * complementCos
    ycomplementCos = y * complementCos
    zcomplementCos = z * complementCos
    xycomplementCos = x * ycomplementCos
    yzcomplementCos = y * zcomplementCos
    zxcomplementCos = z * xcomplementCos
    rot = torch.zeros((axisangle.shape[0], 4, 4)).to(device=axisangle.device)
    rot[:, 0, 0] = torch.squeeze(x * xcomplementCos + cosAngle)
    rot[:, 0, 1] = torch.squeeze(xycomplementCos - zs)
    rot[:, 0, 2] = torch.squeeze(zxcomplementCos + ys)
    rot[:, 1, 0] = torch.squeeze(xycomplementCos + zs)
    rot[:, 1, 1] = torch.squeeze(y * ycomplementCos + cosAngle)
    rot[:, 1, 2] = torch.squeeze(yzcomplementCos - xs)
    rot[:, 2, 0] = torch.squeeze(zxcomplementCos - ys)
    rot[:, 2, 1] = torch.squeeze(yzcomplementCos + xs)
    rot[:, 2, 2] = torch.squeeze(z * zcomplementCos + cosAngle)
    rot[:, 3, 3] = 1
    return rot

def getTranslationMatrix(translation):
    T = torch.zeros(translation.shape[0], 4, 4).to(device=translation.device)
    t = translation.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T

def transformParameters(axisangle, translation, invert=False):
    rotation = rotationFromAxisAngle(axisangle)
    trans = translation.clone()
    if invert:
        rotation = rotation.transpose(1, 2)
        trans *= -1
    T = getTranslationMatrix(trans)
    if invert:
        M = torch.matmul(rotation, T)
    else:
        M = torch.matmul(T, rotation)
    return M

def dispToDepth(disp, minDepth, maxDepth):
    minDisp = 1 / maxDepth
    maxDisp = 1 / minDepth
    scaledDisp = minDisp + (maxDisp - minDisp)*disp
    depth = 1 / scaledDisp
    return scaledDisp, depth

def normalizeImage(image):
    maxValue = float(image.max().cpu().data)
    minValue = float(image.min().cpu().data)
    diff = (maxValue - minValue) if maxValue != minValue else 1e5
    return (image - minValue)/diff
