# AdaptationSeg

This is the Python reference implementation of AdaptionSeg proposed in "Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes".

<pre>
Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes
<a href='https://yangzhang4065.github.io/'>Yang Zhang</a>; Philip David; <a href='http://boqinggong.info/'>Boqing Gong</a>;
International Conference on Computer Vision, 2017
</pre>

<pre>
A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes
<a href='https://yangzhang4065.github.io/'>Yang Zhang</a>; Philip David;  <a href='http://www.cs.ucf.edu/~foroosh/'>Hassan Foroosh</a>; <a href='http://boqinggong.info/'>Boqing Gong</a>;
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019
</pre>

[[**TPAMI paper**]](https://arxiv.org/pdf/1812.09953.pdf) [[**ICCV paper**]](https://github.com/YangZhang4065/AdaptationSeg/raw/master/pdf/ICCV_version.pdf) [[**ArXiv Extended paper**]](https://arxiv.org/abs/1707.09465) [[**Poster**]](https://github.com/YangZhang4065/AdaptationSeg/raw/master/pdf/ICCV2017-poster-v2.pdf)

## [New] Survey of domain adaptation for semantic segmentation

Check out our new survey of domain adaptation for semantic segmentation in our [TPAMI paper](https://arxiv.org/pdf/1812.09953.pdf#page=16).

![Review](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/review.jpg)

## Overview

![Qualitative Results](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/qualitative_results.png)

We introduced a set of constraints to domain-adapt an arbitrary segmentation convolutional neural network (CNN) trained on source domain (synthetic images) to target domain (real images) without accessing target domain annotations.

![Overview](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/overview_cropped-1.png)

## Prerequisites
* Linux
* A CUDA-enabled NVIDIA GPU; Recommend video memory >= 11GB


## Getting Started

### Installation
The code requires following dependencies:
* Python 2/3
* Theano ([installation](http://deeplearning.net/software/theano/install_ubuntu.html))
* Keras>=2.0.5 (Lower version might encounter `Conv2DTranspose` problem with Theano backend) ([installation](https://keras.io/#installation); You might want to install though `pip` since `conda` only offers Keras<=2.0.2)
* Pillow ([installation](https://pillow.readthedocs.io/en/latest/installation.html))

### Keras backend setup

**Make sure your Keras's  `image_data_format` is `channels_first`**.
It is recommended to use `Theano` as the `backend`. However `Tensorflow` should also be okay. Note that using `Tensorflow` will result in lower initial/baseline model performance because the baseline model was trained using Theano.

[How do I check/switch them?](https://keras.io/backend/)


### Download dataset

1, Download `leftImg8bit_trainvaltest.zip` and `leftImg8bit_trainextra.zip` in CityScape dataset [here](https://www.cityscapes-dataset.com/login/). (Require registration)

2, Download `SYNTHIA-RAND-CITYSCAPES` in SYNTHIA dataset [here](http://synthia-dataset.net/downloads/).

3, Download our auxiliary pre-inferred target domain properties (Including both superpixel landmark and label distribution described in the paper) & parsed annotation [here](http://crcv.ucf.edu/data/adaptationseg/ICCV_dataset.zip).

4, Download the submodule [cityscapesScripts](https://github.com/YangZhang4065/cityscapesScripts/tree/c6d717acf7c57b51bd655f3739f3fbf00c7072f2) for evaluation purpose.

5, Unzip and organize them in this way:

```shell
./
├── train_val_DA.py
├── ...
├── cityscapesScripts/
│   ├── ...
│   └── cityscapesscripts/
│       ├── ...
│       └── evaluation/...
└── data/
    ├── Image/
    │   ├── CityScape/           # Unzip from two CityScape zips
    │   │   ├── test/
    │   │   ├── train/
    │   │   ├── train_extra/
    │   │   └── val/
    │   └── SYNTHIA/             # Unzip from the SYNTHIA dataset
    │       └── train/
    ├── label_distribution/      # Unzip from our auxiliary dataset
    │   └── ...
    ├── segmentation_annotation/ # Unzip from our auxiliary dataset
    │   └── ...
    ├── SP_labels/               # Unzip from our auxiliary dataset
    │   └── ...
    └── SP_landmark/             # Unzip from our auxiliary dataset
        └── ...
```
(Hint: If you have already downloaded the datasets but do not want to move them around, you may want to create some [symbolic links](https://kb.iu.edu/d/abbe) of exsiting local datasets)

### Training

Run `train_val_FCN_DA.py` either in your favorite Python IDE or the terminal by typing:

```shell
python train_val_FCN_DA.py
```

This would train the model for six epochs and save the best model during the training. You can stop it and continue to the evaluation during training if you feel it takes too long, however, performance would not be guaranteed then.

### Evaluation

After running `train_val_FCN_DA.py` for at least 500 steps, run `test_FCN_DA.py` either in your favorite Python IDE or the terminal by typing:

```shell
python test_FCN_DA.py
```

This would evaluate both pre-trained SYNTHIA-FCN and adapted FCN over CityScape dataset and print both mean IoU.

## Note
The original framework was implemented in Keras 1 with a custom transposed convolution ops. The performance might be slightly different from the ones reported in the paper. Also, some new commits in TF/Theano optimizer implementation after the code release has broken the losses' numerical stability. I have changed code's optimizer to SGD despite the original paper used Adadelta. You are welcome to try Adadelta/Adam however it seems that they will result in a NaN loss right after training starts. If the NaN problem persists, try to remove the label distribution loss from the training.

## Citation
Please cite our paper if this code benefits your reseaarch:
```
@InProceedings{Zhang_2017_ICCV,
author = {Zhang, Yang and David, Philip and Gong, Boqing},
title = {Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes},
booktitle={The IEEE International Conference on Computer Vision (ICCV)},
volume={2},
number={5},
pages={6},
month = {Oct},
year = {2017}
}

@ARTICLE{Zhang_2019_TPAMI,
author={Zhang, Yang and David, Philip and Foroosh, Hassan and Gong, Boqing},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes},
year={2019},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2019.2903401},
ISSN={1939-3539},
month={},}
```
