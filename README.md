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

| Model                        | Additions                                                                  | Link                                                                                                              |
|------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| CAMLESS                      | Learnable Camera Intrinsics                                                | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/CAMLESS.zip )                   |
| ESPCN                        | Using ESPCN for Upsampling                                                 | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/ESPCN.zip )                     |
| CAMLESS_WEATHER_AUGMENTATION | CAMLESS with weather augmentation                                          | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/CAMLESS_WEATHER_AUG.zip )       |
| MASKCAMLESS                  | Semantic segmentation suggestion from pretrained MASK-RCNN Model + CAMLESS | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS.zip )               |
| MASKCAMLESS_V2               | MASKCAMLESS + skipping loss adjustment for Smoothness loss                 | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_V2.zip )            |
| MASKCAMLESS_ESPCN            | Mask R-CNN + CAMLESS + ESPCN                                               | [`link`](https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN.zip)             |
| MASKCAMLESS_ESPCN_WEATHER    | MASKCAMLESS_ESPCN + weather augmentation                                   | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN_WEATHER.zip ) |
| MASKCAMLESS_ESPCN_V2         | MASKCAMLESS_ESPCN+ skipping loss adjustment for Smoothness loss            | [ `link` ]( https://storage.googleapis.com/depth-estimation-weights/final_weights/MASKCAMLESS_ESPCN_V2.zip )      |


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
| Paper[2] | resnet50 | UNet | bilinear | &#x2717; | 0.8777 | 0.959 | 0.981 | 0.115  | 0.193 | 4.863 | 0.903 |
| CamLess[5] | resneXt50 | UNet | ESPCN | &#10003; | 0.891 | 0.964 | 0.983 | 0.106  | 0.182  | 4.482 | 0.750 |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.8784 | 0.9654 | 0.9867 | 0.109 | 0.1887 | 4.327 | 0.661 |
| Ours | resnet50 | UNet++[3] | bilinear | &#x2717; | 0.8808 | 0.9607 | 0.9835  | 0.1483 | 0.2372 | 6.000 | 3.709 |
| Ours | convnext-tiny[4] | UNet | bilinear | &#x2717; | **0.9145** | 0.9682 | 0.9852  | **0.09386** | 0.1776 | 3.953 | **0.5298** |
| Ours | convnext-tiny | UNet | ESPCN | &#x2717; | 0.8384 | 0.961 | 0.989  | 0.1224 | 0.1892 | **3.886** | 0.587 |
| Ours | convnext-tiny | UNet++ | ESPCN | &#x2717; | 0.8229 | **0.9751** | **0.9902**  | 0.1234 | 0.1933 | 4.07 | 0.6039 |
| Ours | resnet50 | UNet | bilinear | &#10003; | 0.8752 | 0.9575 | 0.9814  | 0.1125 | 0.1984 | 4.55 | 0.6957 |
| Ours | convnext-tiny | UNet | bilinear | &#10003; | 0.7346 | 0.8911 | 0.9491  | 0.1828 | 0.2981 | 7.515 | 1.474 |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.9111 | 0.9733 | 0.9878  | 0.1005 | **0.1693** | 3.978 | 0.5615 |

### Sample Output

|Monodepth2 Output          |ConvNeXt-UNet Output     |
|---------------------------|-------------------------|
|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/fig6.png)|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/WSP-2UP4_pred_convnext-unet_espcn-False.jpg)|

|ConvNeXt-UNet-ESPCN Output |ConvNeXt-UNet++-ESPCN Output |
|---------------------------|-----------------------------|
|![](https://github.com/GuangjieYu1/DL_FinalProject/blob/main/WSP-2UP4-convnext-unet-espcn.jpeg)|![](https://github.com/mayankpoddar/GuangjieYu1/DL_FinalProject/blob/main/WSP-2UP4-convnext-unetplusplus-espcn-modified.jpeg)|

