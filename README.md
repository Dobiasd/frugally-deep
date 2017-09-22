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

Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save(...)`.

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


Requirements and Installation
-----------------------------

A **C++14**-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.6 and Visual C++ 2015

todo add installation like in fplus


todo
----

new structure:

  class Layer
    ctor(...)
    pure virtual apply : [Tensor3] -> [Tensor3]

  class Node
    inbound_nodes : [(node_id, tensor_idx)]
    outbound_layer : SharedPtr layer
    outputs : maybe [Tensor3]
    get_output : tensor_idx -> Tensor3
    set_outputs : [Tensor3] -> () # only for nodes of input-layers?

  class Model : public Layer
    node_pool : Dict Node_id Node
    outputs : [Node_id]
    inputs : [Node_id]

run:
  model.apply : [Tensor3] -> [Tensor3]
    call set_output on all input nodes
    ouput_nodes
      |> transform get_output
      |> concat

add travis

ist upconv (conv transpose) fertig?

typedefs.h nach config.h umbenennen und globals da rein

use cmake with doctests (see fplus)

add 1D layers: Conv, MaxPool, AvgPool, Upsampling

add merge layers (https://keras.io/layers/merge/)

test keras export with different backend

github project description: Use Keras models in C++ with this small header-only library.

add github project tags

float_t sollte float32 sein

remove vscode directory

test sequential as single model

klassiker aus keras als examples

rename all layers to keras names (e.g. unpool -> upsampling2d)

member-funktionen frei machen wo geht

write something about contributing to the project

alles ausser load_model und model in internal namespace verschieben

inline vor alle freien Funktionen

class model_layer machen, model dann nur predict, sonst nix

model hat modellayer als shared_ptr, damit shallow copy geht

predict macht cache auf, der durchgereicht wird, jede node hat id

wie kann man den cache vorzeitig bischen leer machen um RAM zu sparen?

test strides != 1

test paddings valid and same with non-fitting shapes

mention in README that it is always channels_first internally

mention in README that everything is a tensor3, dense eg is (1,1,n)

mention in README that model needs to flatten before dense

geht SeparableConv2D schon?

local response normalization layer https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/

ein layer kann seine output_shape also nur dynamisch berechnen wenn er die input-shape gesagt bekommt. dense asserted dann halt auf input

rename fully_connected_layer to dense_layer

rename matrix to tensor

namespace fd -> fdeep

float_t als template-parameter

readme: tensorflow only tested bisher

travis wie fplus, auch mit warnings und so

size in shape umbenennen

json: CBOR fuer weights und biases? oder in base64 oder sowas?

padding layer implementieren und testen

mention tests in readme (How do I know the results are the same?)