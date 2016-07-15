frugally-deep
=============

* **is a very simplistic deep learning framework written in C++.**
* supports the creation and training of convolutional neural networks.
* is a header-only library.
* only has one dependency ([FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus))
* can use multiple CPUs.
* does not make use of GPUs.
* is quite slow.
* is merely intended to be a personal learning project.
* possibly has some unknown bugs.
* should probably not be used for anything serious. ;-)



todo
----

namespace fd
layer und netz muessen gleiche basisklasse haben
is memory arrangement in matrix good for speed? (cache locallity)
typedefs.h wo value_t als float drin steht
Opencv in own header file
different paddings
different steps

layers:
conv
pooling layer (template max or average)
transposed convolution layer aka deconv
unpooling layer
activation functions (identity, relu, sigmoid) als layer-template
fully connected layer
loss functions: softmax, sigmoid cross entropy, euclidian (l2)

json rausschreiben zwischendurch (temp, rename), html viewer dazu
start one async task per filter in layer? (fplus::execute_paralelly?)

watch again: https://www.youtube.com/watch?v=ue4RJdI8yRA
mini-batches
Vanishing gradient problem?
Weight initialization
regularization? (dropout, prevent overfitting? not needed for autoencoder)
learning momentum?
bias fuer layer, filter oder neurons?
evolutionary optimization instread of backprop? (recombination?)

Affine layer- flow layer?
or instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html
Local Response Normalization layer?

Skip connection aka computational graph?
Caffee import?
Memoization fuer wenn sich irgendwas gar nicht mehr veraendert?


### ideas/goals

image autoencoder

image classification

video autoencoder? (Spatio-temporal? would mean Matrix4d)

Video compression
zweites video dabei, was die differenzframes drin hat
anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video
oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf