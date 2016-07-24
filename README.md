frugally-deep
=============

**is just a bunch of random code, intended solely for my personal learning pleasure.**

The goal is that someday the following things will be true:

![logo](logo/frugally_deep.jpg)

**frugally-deep**

* **is a header-only deep learning framework written in C++.**
* is frugal because it is minimalistic and tries to consume not so much memory.
* supports the creation and training of convolutional neural networks.
* only has one dependency ([FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus))
* can use multiple CPUs.
* is quite slow nonetheless.
* does not make use of GPUs.
* is a needlessly reinvented wheel.
* possibly has some unknown bugs.
* is poorly documented.
* should probably not be used for anything serious.
* is merely intended to be a personal learning project.




todo
----

flatten nestes multilayer nets? Ideally directly create them that way

write inline in front of all free functions

clean up training.h

implement backprop (http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/ , http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf , http://cs231n.github.io/optimization-2/)
give param-deriv out, then something like bfgs can be used
confirm via numerical gradient checking

use cmake with google tests (see fplus)
convolution: different paddings, different steps

layers to implement:
transposed convolution layer aka deconv aka unconv aka fractional step conv
loss layers additionally to sigmoid cross entropy?: Multiclass Support Vector Machine loss (SVM) manhattan (l1)?, euclidian (l2)?

mini-batches

split a validation set from the training set to optimize the hyperparameters

Weight initialization strategies

regularization? (Weights square sum penalty, dropout, prevent overfitting? not needed for autoencoder?)
rule of thumb: 10 times more training examples than parameters

learning momentum?

evolutionary optimization instread of backprop? (momentum mutations? recombination?)
particle swarm optimization? verteilen, gute habe gravitation, andere fliegen dahin, momentum, random speed changes annealing, best reproduces with mutation and worst dies

json rausschreiben zwischendurch (losses, weight distribution) (temp, rename), html viewer dazu
visualize layer and filters as images
start one async task per filter in layer? (fplus::execute_paralelly?)

Affine layer- flow layer?
or instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html
Local Response Normalization layer?

Skip connection aka computational graph?
Caffee import?
Memoization fuer wenn sich irgendwas gar nicht mehr veraendert?


### ideas/goals

image autoencoder

image classification

neural style

video autoencoder? (Spatio-temporal? would mean matrix4d)

Video compression
zweites video dabei, was die differenzframes drin hat
anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video
oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf