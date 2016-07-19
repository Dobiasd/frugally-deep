frugally-deep
=============

**is just a bunch of random code, indended for my personal learning pleasure.**

The goal is that someday the following things will be true:

![logo](logo/frugally_deep.jpg)

**frugally-deep**

* **is a very simplistic deep learning framework written in C++.**
* supports the creation and training of convolutional neural networks.
* is a header-only library.
* only has one dependency ([FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus))
* can use multiple CPUs.
* does not make use of GPUs.
* is quite slow.
* possibly has some unknown bugs.
* is a badly reinvented wheel.
* should probably not be used for anything serious.
* is merely intended to be a personal learning project.




todo
----

implement backprop
bias for which layer types?
let all layers check if input matches input_size and if output matches output_size
would it be better if fc layers had their values along the z axis and not the x axis in the matrix3d?
unique_ptr statt shared_ptr
is memory arrangement in matrix good for speed? (cache locallity)
use cmake with google tests (see fplus)
different paddings
different steps

layers:
transposed convolution layer aka deconv
fully connected layer
loss functions: sigmoid cross entropy, manhattan (l1), euclidian (l2)

watch again: https://www.youtube.com/watch?v=ue4RJdI8yRA
mini-batches
Vanishing gradient problem? ReLU hilft
Weight initialization
regularization? (dropout, prevent overfitting? not needed for autoencoder)
learning momentum?
bias fuer layer, filter oder neurons?

evolutionary optimization instread of backprop? (momentum mutations? recombination?)
particle swarm optimization? verteilen, gute habe gravitation, andere fliegen dahin, momentum, random speed changes annealing, best reproduces with mutation and worst dies

json rausschreiben zwischendurch (losses, weight distribution) (temp, rename), html viewer dazu
start one async task per filter in layer? (fplus::execute_paralelly?)

Affine layer- flow layer?
or instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html
Local Response Normalization layer?
batch normalization (as layer?) (zero mean/unit variance)

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