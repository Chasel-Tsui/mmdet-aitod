# TODbox (Tiny Object Detection Box)
Official implementation of the paper: Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark (accepted by ISPRS J P & RS)

The annotations of **AI-TOD-v2**: https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing

The source code of the **NWD-RKA** please refer to the folder [mmdet-nwdrka](https://github.com/Chasel-Tsui/mmdet-aitod/tree/main/mmdet-nwdrka). 

## Introduction
The Normalized Wasserstein Distance and the RanKing-based Assigning strategy (NWD-RKA) for tiny object detection. 
![demo image](figures/nwdrka.PNG)

A comparison between AI-TOD and AI-TOD-v2.
![demo image](figures/fps2.gif)

## Supported Methods
Supported baselines for tiny object detection:
- [x] [Baselines](mmdet-nwdrka/configs_nwdrka/baseline)

Supported horizontal tiny object detection methods:
- [x] [DotD](mmdet-nwdrka/configs_nwdrka/nwd_rka) [[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Xu_Dot_Distance_for_Tiny_Object_Detection_in_Aerial_Images_CVPRW_2021_paper.html)
- [x] [NWD-RKA](mmdet-nwdrka/configs_nwdrka/nwd_rka)


## Installation and Get Started

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install TODbox:

Note that our TODbox is based on the [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Chasel-Tsui/mmdet-aitod.git
cd mmdet-nwdrka
pip install -r requirements/build.txt
python setup.py develop
```

## Citation

If you use this repo in your research, please consider citing these papers.

```
@inproceedings{xu2021dot,
  title={Dot Distance for Tiny Object Detection in Aerial Images},
  author={Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1192--1201},
  year={2021}
}

@inproceedings{NWDRKA_2020_ISPRS,
    title={Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark},
    author={Chang, Xu and Jinwang, Wang and Wen, Yang and Huai, Yu and Lei, Yu and Gui-Song, Xia},
    booktitle=ISPRS Journal of Photogrammetry and Remote Sensing,
    pages={In press},
    year={2022},
}
```

## References
* [AI-TOD](https://github.com/jwwangchn/AI-TOD)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [DOTA](https://captain-whu.github.io/DOTA/index.html)



