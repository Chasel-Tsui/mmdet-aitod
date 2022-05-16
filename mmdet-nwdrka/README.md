## Installation and Get Started

Required environments
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


Install TODbox

Note that our TODbox is based on the [MMDetection2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Chasel-Tsui/mmdet-aitod.git
cd mmdet-nwdrka
pip install -r requirements/build.txt
python setup.py develop
```

## Performance
Please refer to the paper.

For your convenience, we also provide the performance of the model trained on **AI-TOD-v2 train set** and validated on the **AI-TOD-v2 val set**. 

Model | AP | $AP_{50}$ | AP~75~ |AP~vt~ | AP~t~ | AP~s~ | AP~m~ | AR^1500^
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
FR | 12.9 | 29.5 | 9.2 | 0.0 | 9.5 | 27.3 | 37.2 | 19.5
FR w/ NWD-RKA | 21.9 | 51.8 | 13.9 | 5.8 | 21.8 | 27.3 | 37.8 | 34.6
