# THIS IS ALL UNFINISHED WORK IN PROGRESS. Please do not try to use it (yet). ;-)

![logo](logo/fdeep.png)

[![Build Status](https://travis-ci.org/Dobiasd/frugally-deel.svg?branch=master)][travis]
[![(License MIT 1.0)](tps://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[travis]: https://travis-ci.org/Dobiasd/frugally-deep
[license]: LICENSE


frugally-deep
=============
**let's you run your Keras models in C++.**


Table of contents
-----------------
  * [Introduction](#introduction)
  * [Requirements and Installation](#requirements-and-installation


Introduction
------------

Do you want to use your already-trained Keras models in C++? And do you want to avoid linking against a huge library like TensorFlow? Then frugally-deep is exactly for you.

**frugally-deep**

* **is a small header-only library without external dependencies.**
* supports not only [sequential models](https://keras.io/getting-started/sequential-model-guide/) but also more complex computational graphs created with the [functional API](https://keras.io/getting-started/functional-api-guide/).
* uses only the CPU (single threaded).


### Supported layer types

* AveragePooling2D
* BatchNormalization
* Conv2D
* Dense
* ELU
* Flatten
* LeakyReLU
* MaxPooling2D
* ReLU
* Sigmoid
* Softmax
* Softplus
* Tanh
* UpSampling2D


Requirements and Installation
-----------------------------

A **C++11**-compatible compiler is needed (C++14 if you want to use the features from `namespace fwd`). Compilers from these versions on are fine: GCC 4.9, Clang 3.6 and Visual C++ 2015





todo
----

ist upconv (conv transpose) fertig?

typedefs.h nach config.h umbenennen und globals da rein

use cmake with google tests (see fplus)

Affine layer- flow layer?
or instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html
Local Response Normalization layer?

klassiker aus keras als examples