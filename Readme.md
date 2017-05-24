# Neural style

## Introduction
This repository contains a Keras (theano backend) implementation of "A Neural Algorithm of Artistic Style" by L. Gatys, A. Ecker, and M. Bethge. The paper proposes that one can separate the content and the style of an image, and futhur presents a method for transferring the artistic style of one input image onto another.

The implementation include techniques in the original paper as well as more recent variations described in the following papers:

-  ["A Neural Algorithm of Artistic Style"](http://arxiv.org/abs/1508.06576) by L. Gatys, A. Ecker, and M. Bethge.


This is an exmaple of neural style transfer with a starry night style to a bengal tiger.

<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/bengaltiger.png" width="512"/>
<img src="https://s3-us-west-2.amazonaws.com/temptosync/content_style.png"width="290"/>

The following results are consistent with the results in the [original paper](http://arxiv.org/abs/1508.06576).

<div align="center">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/neuralstyletransfer/tubingen.png" height="250px">


<img src="https://s3-us-west-2.amazonaws.com/temptosync/neuralstyletransfer/shipwreck.png" height="250px">

<img src="https://s3-us-west-2.amazonaws.com/temptosync/starrynight8.png" height = "250px"
width = "317px">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/neuralstyletransfer/scream.png" height="250px">

<img src="https://s3-us-west-2.amazonaws.com/temptosync/neuralstyletransfer/seatednude.png" height="250px">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/neuralstyletransfer/compositionvii.png" height="250px">
</div>


## Running

```
python neural_style.py --content <content file> --styles <style file> --output<output file>
```

## Experiments

To see the effect of various style layers on the results, I created these images by using only a single style layer. Based on the results below, to have a strong style, one can increase the weight for conv2_1, conv3_1 and conv4_1. To have a weaker style, one can increase the weight for conv1_1 and conv5_1. For comparison, the bengal tiger above has a weight average of all five layers with each layer contributing 20% of the style loss function.

<div align="center">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/l1.png" width = "160">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/l2.png" width ="160">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/l3.png" width = "160">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/l4.png" width ="160">
<img src="https://s3-us-west-2.amazonaws.com/temptosync/l5.png" width = "160">
</div>
*Rows*: image created using only a single style layer, respectively 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'.  
## optimization
Neural net operations are handled by Keras with Tensorflow as backend, while loss minimization and other miscellaneous matrix operations are performed using numpy and scipy. L-BFGS is used for minimization.


## Implementation Details
All images were rendered on a machine with:
* **CPU:** Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
* **GPU:** Telsa K80
* **OS:** Linux Ubuntu 16.04.1 LTS 64-bit
* **CUDA:** 8.0
* **python:** 3.5
* **keras:** 1.2.2

## Acknowledgements
- bengal tiger photo from [National Geographic website](http://www.nationalgeographic.com/animals/mammals/b/bengal-tiger/)

The implementation is based on the projects:
* Keras implementation 'neural-style' by
[Jeremy Howard](http://files.fast.ai/part2/lesson8/neural-style.ipynb)
* Torch (Lua) implementation 'neural-style' by [jcjohnson](https://github.com/jcjohnson)
* Tensorflow implementation by
[cysmith](https://github.com/cysmith/neural-style-tf)

## Citation

If you find this code useful for your research, please cite:

```
@misc{Li2017,
  author = {Li, Xinxin},
  title = {neural-style-keras},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cysmith/neural-style-tf}},
}
```
