# AdaptationSeg

This is the Python reference implementation of AdaptionSeg proposed in "Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes".

<pre>
Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes
<a href='https://yangzhang4065.github.io/'>Yang Zhang</a>;Phillip David;<a href='http://crcv.ucf.edu/people/faculty/Gong/'>Boqing Gong</a>;
International Conference on Computer Vision, 2017
</pre>

![Qualitative Results](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/qualitative_results.png)

We proposed a set of constraints to domain-adapt an arbitrary segmentation convolutional neural network (CNN) trained on source domain (synthetic images) to target domain (real images) without accessing target domain annotations.

![Overview](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/overview_cropped-1.png)

## Prerequisites
* Linux
* A CUDA-enabled NVIDIA GPU; Recommend video memory >= 11GB


## Getting Started

### Installation
The code requires following dependencies:
* Python 2/3
* Theano ([installation](http://deeplearning.net/software/theano/install_ubuntu.html))
* Keras>=2.0.5 (Lower version might encounter `Conv2DTranspose` problem with Theano backend) ([installation](https://keras.io/#installation); You might want to install though `pip` as `conda` only offers Keras<=2.0.2)
* Pillow ([installation](https://pillow.readthedocs.io/en/latest/installation.html))

### Keras backend setup

**Make sure your Keras `backend` is `Theano` and `image_data_format` is `channels_first`**

[How do I check/switch them?](https://keras.io/backend/)


### Download dataset

1, Download `leftImg8bit_trainvaltest.zip` and `leftImg8bit_trainextra.zip` in CityScape dataset [here](https://pillow.readthedocs.io/en/latest/installation.html). (Require registration)

2, Download `SYNTHIA-RAND-CITYSCAPES` in SYNTHIA dataset [here](http://synthia-dataset.net/download-2/).

3, Download our auxiliary pre-inferred target domain properties (Including both superpixel landmark and label distribution described in the paper) & parsed annotation [here](http://crcv.ucf.edu/data/adaptationseg/ICCV_dataset.zip).

4, Unzip and organize them in this way:

```shell
./
├── train_val_DA.py
├── ...
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

### Training

Run `train_val_FCN_DA.py` either in your favorite Python IDE or in terminal by typing:

```shell
python train_val_FCN_DA.py
```

This would train the model for 6 epochs and save the best model during the training. You can stop it and continue to the evaluation during training if you feel it takes too long however performance would not be guaranteed then.

### Evaluation

After running `train_val_FCN_DA.py` for at least 500 steps, run `test_FCN_DA.py` either in your favorite Python IDE or in terminal by typing:

```shell
python test_FCN_DA.py
```

This would evaluate both pre-trained SYNTHIA-FCN and adapted FCN over CityScape dataset and print both mean IoU.

## Note
The original framework was implmented in Keras 1 with a custom transposed convolution ops. The performance might be slightly different from the ones reported in the paper.
