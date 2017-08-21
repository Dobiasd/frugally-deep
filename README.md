# THIS IS ALL UNFINISHED WORK IN PROGRESS. Please do not try to use it (yet). ;-)

![logo](logo/fdeep.png.hidden)

[![Build Status](https://travis-ci.org/Dobiasd/frugally-deel.svg?branch=master)][travis]
[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[travis]: https://travis-ci.org/Dobiasd/frugally-deep
[license]: LICENSE


frugally-deep
=============
**Use Keras models in C++ with this small header-only library.**


Table of contents
-----------------
  * [Introduction](#introduction)
  * [Usage](#usage)
  * [Requirements and Installation](#requirements-and-installation


Introduction
------------

Would you like to use your already-trained Keras models in C++? And do you want to avoid linking against a huge library like TensorFlow? Then frugally-deep is exactly for you.

**frugally-deep**

* **is a small header-only library without external dependencies.**
* supports inference (`model.predict`) not only [sequential models](https://keras.io/getting-started/sequential-model-guide/) but also computational graphs with a more complex topology, created with the [functional API](https://keras.io/getting-started/functional-api-guide/).
* uses only the CPU (single threaded).


### Supported layer types

* AveragePooling2D
* BatchNormalization
* Concatenate
* Conv2D
* Dense
* ELU
* Flatten
* LeakyReLU
* MaxPooling2D
* ReLU
* SeLU
* Sigmoid
* Softmax
* Softplus
* Tanh
* UpSampling2D


### Also supports

* multiple inputs and outputs
* nested models
* residual connections
* shared layers


### Layer types not supported yet

* Conv3D
* Custom layers
* Cropping*D
* Embedding layers
* Global*Pooling
* Lambda
* Layer wrappers (TimeDistributed etc.)
* LocallyConnected*D
* Masking
* Merge layers (add etc.)
* Noise layers (GaussianNoise etc.)
* Permute
* PReLU
* Recurrent layers (LSTM etc.)
* Reshape
* SeparableConv2D
* ThresholdedReLU


Usage
-----

todo

Save your model to a single HDF5 file using `model.save(...)` in Python.

export keras model

c++ example code



Requirements and Installation
-----------------------------

A **C++11**-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.6 and Visual C++ 2015





todo
----

concat layer

add travis

add installation like in fplus

ist upconv (conv transpose) fertig?

typedefs.h nach config.h umbenennen und globals da rein

use cmake with doctests (see fplus)

klassiker aus keras als examples

add 1D layers: Conv, MaxPool, AvgPool, Upsampling

add merge layers (https://keras.io/layers/merge/)

test keras export with different backend

github project description: Use Keras models in C++ with this small header-only library.

add github project tags