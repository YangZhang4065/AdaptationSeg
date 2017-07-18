# AdaptationSeg

This is the Python reference implementation for AdaptionSeg proposed in "Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes".

<pre>
Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes
<a href='https://yangzhang4065.github.io/'>Yang Zhang</a>;Phillip David;<a href='http://crcv.ucf.edu/people/faculty/Gong/'>Boqing Gong</a>;
International Conference on Computer Vision, 2017
</pre>

![Qualitative Results](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/qualitative_results.png)

We proposed a set of contraints to domain-adapt an arbitrary segmentation convolutional neural network (CNN) trained on source domain (sythetic images) to target domain (real images) without accessing target domain annotations.

![Overview](https://github.com/YangZhang4065/AdaptationSeg/blob/master/imgs/overview_cropped-1.png)

## Prerequisites
* Python 3
* [Theano](http://deeplearning.net/software/theano/)
* [Keras](https://keras.io/)>=2.0.5 (Lower version might encounter `Conv2DTranspose` problem with Theano backend)
* [Pillow](https://python-pillow.org/)

## Usage
**Make sure your Keras `backend` is `Theano` and `image_data_format` is `channels_first`**

[How do I check/change them?](https://keras.io/backend/)

## Note
The original framework was implmented in Keras 1 with a custom transposed convolution ops. The performance might be slightly different from the ones reported in the paper.
