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

* **is a small header-only library without external dependencies.**
* is very easy to integrate and use
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
    output : maybe [Tensor3]
    get_output : tensor_idx -> Tensor3
    set_output : [Tensor3] -> () # only for nodes of input-layers?

  class model : public layer
    nodes : Dict node_id, node
    outputs : [node_id]
    inputs : [node_id]

run:
  model.predict : [Tensor3] -> [Tensor3]
    call set_output on all input nodes
    ouput_nodes
      |> transform get_output
      |> concat

model::get_output(tensor_idx) -> output[tensor_idx - 1]
https://stackoverflow.com/questions/46011749/understanding-keras-model-architecture-node-index-of-nested-model

concat layer

add travis

ist upconv (conv transpose) fertig?

typedefs.h nach config.h umbenennen und globals da rein

use cmake with doctests (see fplus)

klassiker aus keras als examples

add 1D layers: Conv, MaxPool, AvgPool, Upsampling

add merge layers (https://keras.io/layers/merge/)

test keras export with different backend

github project description: Use Keras models in C++ with this small header-only library.

add github project tags