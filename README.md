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

layer und netz muessen gleiche basisklasse haben
Memoization fuer wenn sich irgendwas gar nicht mehr veraendert?
mini-batches
json rausschreiben zwischendurch (temp, rename), html viewer dazu
start one async task per filter in layer? (fplus::execute_paralelly?)
memory arrangement in matrix good for speed? (cache locallity)
Vanishing gradient
Weight initialization
typedefs.h wo value_t als float drin steht
activation functions (identity, relu, sigmoid) als layer-template
regularization (dropout)?
loss calculation: softmax, euclidian
learning momentum?
local response normalization
die eine Vorlesung nochmal gucken
Caffee import?
Opencv in own header file



namespace fd
transposed convolution layer aka deconv
pooling layer (template max or average)
unpooling layer
relu layer and other activation layers (activation func as templ param?)
fully connected layer
Affine layer- flow layer?
Local Response Normalization layer?
loss als layer? softmax, sigmoid cross entropy, euclidian (l2)
bias fuer layer, filter oder neurons?
different paddings
different steps

Skip connection aka computational graph?

evolutionary optimization instread of backprop?

regularization (prevent overfitting? not needed for autoencoder)

Spatio-temporal video autoencoder? would mean Matrix4d

zweites video dabei, was die differenzframes drin hat
anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video
oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf

instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html

goals:
image autoencoder
image classification
video autoencoder? (perhaps)
