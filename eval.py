import sys
import os
import torch
import numpy as np
from collections import OrderedDict
from Trainer import Trainer
from utils import dispToDepth

def usage():
    print("python eval.py /path/to/config.yaml /path/to/unzipped/weights/")

args = sys.argv
if len(args) != 3:
    usage()

config_path = args[1]
weights_path = args[2]
t = Trainer(config_path)
device = t.device
modelName = t.modelName

# Encoder Model
encoder = t.models["encoder"]
encoderDict = torch.load(os.path.join(weights_path, "encoder.pth"), map_location=device)
try:
    height = encoderDict.pop("height")
    width = encoderDict.pop("width")
    use_stereo = encoderDict.pop("use_stereo")
except:
    pass
encoder.load_state_dict(encoderDict)
encoder.to(device)
encoder.eval() 

# Depth Decoder Model
decoder = t.models["decoder"]
try:
    decoder.load_state_dict(torch.load(os.path.join(weights_path, "decoder.pth"), map_location=device))
except:
    odict = torch.load(os.path.join(weights_path, "depth.pth"), map_location=device)
    odict_compat = OrderedDict([(key.replace("conv.conv", "conv"), value) for key, value in odict.items()])
    del odict
    decoder.load_state_dict(odict_compat)
decoder.to(device)
decoder.eval()

abs_rel = []
sq_rel = []
rms = []
log_rms = []
a1 = []
a2 = []
a3 = []

for batch_idx, inputs in enumerate(t.valLoader):
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    input_images = inputs[("color", 0, 0)]
    features = encoder(input_images)
    outputs = decoder(features)
    _, depth = dispToDepth(outputs[("disp", 0)], 0.1, 100.0)
    outputs[("depth", 0, 0)] = depth
    losses = t.losses["Depth"](inputs, outputs)
    abs_rel.append(losses["de/abs_rel"].item())
    sq_rel.append(losses["de/sq_rel"].item())
    rms.append(losses["de/rms"].item())
    log_rms.append(losses["de/log_rms"].item())
    a1.append(losses["da/a1"].item())
    a2.append(losses["da/a2"].item())
    a3.append(losses["da/a3"].item())
print("a1 : {}".format(np.mean(a1)))
print("a2 : {}".format(np.mean(a2)))
print("a3 : {}".format(np.mean(a3)))
print("abs_rel : {}".format(np.mean(abs_rel)))
print("sq_rel : {}".format(np.mean(sq_rel)))
print("rms : {}".format(np.mean(rms)))
print("log_rms : {}".format(np.mean(log_rms)))
