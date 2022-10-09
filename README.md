# TODbox (Tiny Object Detection Box)
This is a repository of the official implementation of the following paper: 
* [[Paper]](https://www.sciencedirect.com/science/article/pii/S0924271622001599?dgcid=author)[[Code]](mmdet-nwdrka) Detecting tiny Objects in aerial images: A normalized Wasserstein distance and A new benchmark ([ISPRS J P & RS](https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing), 2022)
* [[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Xu_Dot_Distance_for_Tiny_Object_Detection_in_Aerial_Images_CVPRW_2021_paper.html)[[Code]](mmdet-nwdrka) Dot distance for tiny object detection in aerial images ([CVPRW](http://www.classic.grss-ieee.org/earthvision2021/), 2021)



## Introduction
The Normalized Wasserstein Distance and the RanKing-based Assigning strategy (NWD-RKA) for tiny object detection. 
![demo image](figures/nwdrka.PNG)

A comparison between AI-TOD and AI-TOD-v2.
![demo image](figures/fps2.gif)

## Supported Data
- [x] [AI-TOD](https://github.com/jwwangchn/AI-TOD)
- [x] [AI-TOD-v2](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing)

Notes: The images of the **AI-TOD-v2** are the same of the **AI-TOD**. In this stage, we only release the train, val annotations of the **AI-TOD-v2**, the test annotations will be used to hold further competitions.

## Supported Methods
Supported baselines for tiny object detection:
- [x] [Baselines](mmdet-nwdrka/configs_nwdrka/baseline)

Supported horizontal tiny object detection methods:
- [x] [DotD](mmdet-nwdrka/configs_nwdrka/nwd_rka) 
- [x] [NWD-RKA](mmdet-nwdrka/configs_nwdrka/nwd_rka)
- [] [RFLA] 

Supported rotated tiny object detection methods:
- [] Stay tuned, coming soon!


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
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={1192--1201},
  year={2021}
}

@inproceedings{NWDRKA_2022_ISPRS,
    title={Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark},
    author={Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Huai and Yu, Lei and Xia, Gui-Song},
    booktitle={ISPRS Journal of Photogrammetry and Remote Sensing},
    volume={190},
    pages={79--93},
    year={2022},
}
```

## References
* [AI-TOD](https://github.com/jwwangchn/AI-TOD)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [DOTA](https://captain-whu.github.io/DOTA/index.html)



