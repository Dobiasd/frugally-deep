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
  * [Requirements and Installation](#requirements-and-installation)
  * [Internals](#internals)


Introduction
------------

Would you like to use your already-trained Keras models in C++? And do you want to avoid linking against a huge library like TensorFlow? Then frugally-deep is exactly for you.

**frugally-deep**

* **is a small header-only library.**
* is very easy to integrate and use.
* depends only on [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus) - also a small header-only library
* supports inference (`model.predict`) not only for [sequential models](https://keras.io/getting-started/sequential-model-guide/) but also for computational graphs with a more complex topology, created with the [functional API](https://keras.io/getting-started/functional-api-guide/).
* has a small memory footprint.
* does not make use of a GPU.


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

Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save(...)`. The `image_data_format` in your model shoud be `channels_last`, which is the default when using the Tensorflow backend.

Now convert it to the frugally-deep file format with `keras_export/export_model.py`

Finally load it in C++ (`fdeep::load_model`) and use `model.predict()` to invoke a forward pass with your data.

The following minimal example shows the full workflow:

```python
# create_model.py
import keras
todo example
```

```
python keras_export/export_model.py keras_model.h5 fdeep_model.json
```

```cpp
// main.cpp
#include <fdeep/fdeep.hpp>
const auto model = fdeep::load_model("fdeep_model.json");
const auto result = model.predict({{{1,2,3}}});
```

When using `export_model.py` some test cases are generated automatically and saved along with your model. `fdeep::load_model` runs these tests to make sure the results of a forward pass in frugally-deep are the same as if run in Keras.

todo image example


Requirements and Installation
-----------------------------

A **C++14**-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.7 (libc++ 3.7) and Visual C++ 2015

todo add installation like in fplus



Internals
---------

frugally-deep uses `channels_first` (`(depth/channels, height, width`) as its `image_data_format` internally. `export_model.py` takes care of all necessary conversions.
From then on everything is handled as a tensor with rank 3. Dense layers for example take its input flattened to a shape of `(1, 1, n)`. This is also the shape you will receive as the output of a `softmax` layer for example.





todo
----

add travis

ist upconv (conv transpose) fertig?

use cmake with doctests (see fplus)

add 1D layers: Conv, MaxPool, AvgPool, Upsampling

add merge layers (https://keras.io/layers/merge/)

github project description: Use Keras models in C++ with this small header-only library.

add github project tags

remove vscode directory

klassiker aus keras als examples

write something about contributing to the project

wie kann man den cache vorzeitig bischen leer machen um RAM zu sparen?

test strides != 1

test paddings valid and same with non-fitting shapes

test pooling with non-fitting data

test upsampling2d with non-even factors

geht SeparableConv2D schon?

local response normalization layer https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/

float_t als template-parameter

travis wie fplus, auch mit warnings und so

json: CBOR fuer weights und biases? oder in base64 oder sowas?

padding layer implementieren und testen