# Depth Estimation using Self Supervised learning 


<!-- ABOUT THE PROJECT -->

## Contributors

* [Zhenyu Ma]
* [Guangjie Yu]

## Overview

The goal of this project is to improve the performance of the Monodepth2 model by incorporating new techniques and using a common encoder backbone. We will test our modifications using the KITTI and NYUv2 datasets and consider using various data augmentation techniques. Our ultimate aim is to select the optimal architecture and backbone for the Monodepth2 model.


## Environment Setup
```
conda env create -f depthestimate_env.yaml
conda activate depthestimate_env
```


## Train Model

Training your model
```
python train.py --model MONODEPTH2 --conf configs/model_config.cfg 
```

To run in background

```
nohup python -u train.py --model MONODEPTH2 > output.log &
```

## Experiment

### Result Reported


| Impl | Encoder | Arch | Upsampling | K | a1 | a2 | a3 | abs_rel | log_rms | rms | sq_rel |
|---------------|--------------|-----------------|--------|--------|--------|---------|---------|-------|--------|--------|--------|
| Paper | resnet50 | UNet | bilinear | &#x2717; | 0.8777 | 0.959 | 0.981 | 0.115  | 0.193 | 4.863 | 0.903 |
| CamLess | resneXt50 | UNet | ESPCN | &#10003; | 0.891 | 0.964 | 0.983 | 0.106  | 0.182  | 4.482 | 0.750 |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.8784 | 0.9654 | 0.9867 | 0.109 | 0.1887 | 4.327 | 0.661 |
| Ours | resnet50 | UNet++ | bilinear | &#x2717; | 0.8808 | 0.9607 | 0.9835  | 0.1483 | 0.2372 | 6.000 | 3.709 |
| Ours | convnext-tiny | UNet | bilinear | &#x2717; | **0.9145** | 0.9682 | 0.9852  | **0.09386** | 0.1776 | 3.953 | **0.5298** |
| Ours | convnext-tiny | UNet | ESPCN | &#x2717; | 0.8384 | 0.961 | 0.989  | 0.1224 | 0.1892 | **3.886** | 0.587 |
| Ours | convnext-tiny | UNet++ | ESPCN | &#x2717; | 0.8229 | **0.9751** | **0.9902**  | 0.1234 | 0.1933 | 4.07 | 0.6039 |
| Ours | resnet50 | UNet | bilinear | &#10003; | 0.8752 | 0.9575 | 0.9814  | 0.1125 | 0.1984 | 4.55 | 0.6957 |
| Ours | convnext-tiny | UNet | bilinear | &#10003; | 0.7346 | 0.8911 | 0.9491  | 0.1828 | 0.2981 | 7.515 | 1.474 |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.9111 | 0.9733 | 0.9878  | 0.1005 | **0.1693** | 3.978 | 0.5615 |

### Sample Output

|Monodepth2 Output          |ConvNeXt-UNet Output     |
|---------------------------|-------------------------|
|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/result/fig6.png)|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/result/WSP-2UP4_pred_convnext-unet_espcn-False.jpg)|

|ConvNeXt-UNet-ESPCN Output |ConvNeXt-UNet++-ESPCN Output |
|---------------------------|-----------------------------|
|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/result/WSP-2UP4-convnext-unet-espcn.jpeg)|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/result/WSP-2UP4-convnext-unetplusplus-espcn-modified.jpeg)|

