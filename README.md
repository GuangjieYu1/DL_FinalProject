<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://rehost.in/templates">
    <img src="https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo.gif" alt="testVideo" width="450" height="300">
    <img src="https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-baseline-resnet-unet.gif" alt="baseline" width="450" height="300">
  </a>

<h3 align="center">Depth Estimation using Self Supervised learning</h3>
  <p align="center">
    an extension of "Digging into self-supervised monocular depth estimation"
    <br />
    <a href="https://www.tensorflow.org/tensorboard](https://pytorch.org/"><strong> Pytorch »</strong></a> | <a href="https://www.tensorflow.org/tensorboard"><strong> Tensorboard »</strong></a>
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
# Overview

Keeping Monodepth2[1] as our baseline model, we propose certain architectural changes that
improve the performance of Monodepth V2 by incorporating recent developments for convolutional
neural networks and using a common encoder backbone. In the next phase, we plan to incorporate
NYUv2 dataset and experiment with various augmentation techniques to further improve the
performance on the optimal backbone and architecture selected. All the experiments are performed
on the KITTI dataset [5] and the NYUv2 dataset [6].

* [Environment Setup and Model Training](#env)
* [Experimentation Results](#results)
* [References](#ref)

<p align="right">(<a href="#top">back to top</a>)</p>

<a name="env"></a>

# Environment Setup

1. Install Conda:

```
conda env create -f depthestimate_env.yaml
conda activate depthestimate_env
```

2. Install project dependencies:

```
pip install -r requirements.txt
```

# Train Model

```
python main.py --conf configs/config.yaml 
```

You can run it in the background on HPC using:

```
nohup python main.py --conf configs/config.yaml > output.log &
```
Use tb flag to enable tensorboard
```
python main.py --conf configs/config.yaml -tb 
```
use tbpath `-tbpth ./logs` for custom log path
  
<a name="results"></a>
# Experiment

## Data Reported

| Impl | Encoder | Arch | Upsampling | K | a1 | a2 | a3 | abs_rel | log_rms | rms | sq_rel | Trained Weights |
|---------------|--------------|-----------------|--------|--------|--------|---------|---------|-------|--------|--------|--------| -------- |
| Paper[2] | resnet50 | UNet | bilinear | &#x2717; | 0.8777 | 0.959 | 0.981 | 0.115  | 0.193 | 4.863 | 0.903 | - |
| CamLess[5] | resneXt50 | UNet | ESPCN | &#10003; | 0.891 | 0.964 | 0.983 | 0.106  | 0.182  | 4.482 | 0.750 | - |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.8784 | 0.9654 | 0.9867 | 0.109 | 0.1887 | 4.327 | 0.661 | [Download](https://storage.googleapis.com/depthestimation-weights/baseline-resnet-unet.zip) |
| Ours | resnet50 | UNet++[3] | bilinear | &#x2717; | 0.8808 | 0.9607 | 0.9835  | 0.1483 | 0.2372 | 6.000 | 3.709 | [Download](https://storage.googleapis.com/depthestimation-weights/resnet-unetplusplus.zip) |
| Ours | convnext-tiny[4] | UNet | bilinear | &#x2717; | **0.9145** | 0.9682 | 0.9852  | **0.09386** | 0.1776 | 3.953 | **0.5298** | [Download](https://storage.googleapis.com/depthestimation-weights/convnext-unet.zip) |
| Ours | convnext-tiny | UNet | ESPCN | &#x2717; | 0.8384 | 0.961 | 0.989  | 0.1224 | 0.1892 | **3.886** | 0.587 | [Download](https://storage.googleapis.com/depthestimation-weights/convnext-unet-espcn.zip) |
| Ours | convnext-tiny | UNet++ | ESPCN | &#x2717; | 0.8229 | **0.9751** | **0.9902**  | 0.1234 | 0.1933 | 4.07 | 0.6039 | [Download](https://storage.googleapis.com/depthestimation-weights/convnext-unetplusplus-espcn.zip) |
| Ours | resnet50 | UNet | bilinear | &#10003; | 0.8752 | 0.9575 | 0.9814  | 0.1125 | 0.1984 | 4.55 | 0.6957 | [Download](https://storage.googleapis.com/depthestimation-weights/resnet-unet-camnet.zip) |
| Ours | convnext-tiny | UNet | bilinear | &#10003; | 0.7346 | 0.8911 | 0.9491  | 0.1828 | 0.2981 | 7.515 | 1.474 | [Download](https://storage.googleapis.com/depthestimation-weights/convnext-unet-camnet.zip) |
| Ours | resnet50 | UNet | ESPCN | &#x2717; | 0.9111 | 0.9733 | 0.9878  | 0.1005 | **0.1693** | 3.978 | 0.5615 | [Download](https://storage.googleapis.com/depthestimation-weights/resnet-unet-espcn.zip) |

|Monodepth2 Output          |ConvNeXt-UNet Output     |
|---------------------------|-------------------------|
|![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/fig6.png)|![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/WSP-2UP4_pred_convnext-unet_espcn-False.jpg)|

|ConvNeXt-UNet-ESPCN Output |ConvNeXt-UNet++-ESPCN Output |
|---------------------------|-----------------------------|
|![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/WSP-2UP4-convnext-unet-espcn.jpeg)|![](https://github.com/mayankpoddar/depthestimation/blob/main/assets/WSP-2UP4-convnext-unetplusplus-espcn-modified.jpeg)|

|Sample Video | Monodepth2 Output |
|-------------|-------------------|
|![testVideo](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo.gif)|![baseline](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-baseline-resnet-unet.gif)|

|ConvNeXt-UNet Output | ConvNeXt-UNet++-ESPCN Output |
|---------------------|----------------------------|
|![convnext-unet](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-convnext-unet.gif)|![convnext-unetplusplus-espcn](https://github.com/mayankpoddar/depthestimation/blob/main/predictions/testVideo-convnext-unetplusplus-espcn.gif)|

## Reproduce Results

### Running on Datasets

Unzip your weights to /path/to/unzipped/weights. 
The results shown above can be reproduced by running:

```
python eval.py /path/to/config.yaml /path/to/unzipped/weights/
```

to evaluate any model on KITTI dataset.

### Running on Custom Image and Videos

* `/test-image.ipynb`: This notebook can be used for running experiment on custom images.
* `/test-video.ipynb`: This notebook can be used for running experiment on custom images.

<hr/>

<p align="right">(<a href="#top">back to top</a>)</p>

<a name="ref"></a>
## References

[1] Godard, Cl ́ement, et al., ”Digging into self-supervised monocular depth estimation.” Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019. arXiv:1806.01260

[2] Source Code of Monodepth2: GitHub - nianticlabs/monodepth2: [ICCV 2019] Monocular depth estimation from a single image.

[3] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang, “UNet++: A Nested U-Net Architecture for Medical Image Segmentation”. arXiv:1807.10165.

[4] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, “A Con- vNet for the 2020s”. arXiv:2201.03545.

[5] A. Geiger, P. Lenz, C. Stiller, R. Urtasun, ‘Vision meets Robotics: The KITTI Dataset’, International Journal of Robotics Research (IJRR), 2013.

[6] P. K. Nathan Silberman, Derek Hoiem, R. Fergus, ‘Indoor Segmentation and Support Inference from RGBD Images’, ECCV, 2012.

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgements

Prof. Siddharth Garg and Prof. Arsalan Mosenia supervised this study as part of the ECE-GY 7123:
Intro To Deep Learning Course at New York University. We appreciate NYU providing the team
with High Performance Computing facilities.


## Contributors

* [Mayank Poddar](https://github.com/mayankpoddar)
* [Akash Mishra](https://github.com/akashsky1994)
* [Shikhar Vaish](https://github.com/svr8)
