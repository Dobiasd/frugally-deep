# THIS IS ALL UNFINISHED WORK IN PROGRESS. Please do not try to use it (yet). ;-)

![logo](logo/fdeep.png.hidden)

[![Build Status](https://travis-ci.org/Dobiasd/frugally-deep.svg?branch=master)][travis]
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
  * [Performance](#performance)
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
* utterly ignores even the most powerful GPU in your system and uses only one CPU core. ;-)


### Supported layer types

* Add
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
* SeparableConv2D
* Sigmoid
* Softmax
* Softplus
* Tanh
* UpSampling2D
* ZeroPadding2D


### Also supported

* multiple inputs and outputs
* nested models
* residual connections
* shared layers


### Layer types currently not supported

* Conv2DTranspose
* Conv3D
* Custom layers
* Cropping*D
* Embedding layers
* Global*Pooling
* Lambda
* Layer wrappers (TimeDistributed etc.)
* LocallyConnected*D
* Masking
* Merge layers (sub etc.)
* Noise layers (GaussianNoise etc.)
* Permute
* PReLU
* Recurrent layers (LSTM etc.)
* Reshape
* ThresholdedReLU
* DepthwiseConv2D


Usage
-----

1) Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save(...)`. The `image_data_format` in your model must be `channels_last`, which is the default when using the TensorFlow backend. Models created with a different `image_data_format` and other backends are not officially supported nor tested.

2) Now convert it to the frugally-deep file format with `keras_export/export_model.py`

3) Finally load it in C++ (`fdeep::load_model`) and use `model.predict()` to invoke a forward pass with your data.

The following minimal example shows the full workflow:

```python
# create_model.py
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(4,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
  np.asarray([[1,2,3,4],[2,3,4,5]]),
  np.asarray([[1,0,0], [0,0,1]]), epochs=10)

model.save('keras_model.h5')
```

```
python3 keras_export/export_model.py keras_model.h5 fdeep_model.json
```

```cpp
// main.cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor3(fdeep::shape3(1, 1, 4), {1,2,3,4})});
    std::cout << fdeep::show_tensor3(result) << std::endl;
}
```

When using `export_model.py` a test case (input and corresponding output values) are generated automatically and saved along with your model. `fdeep::load_model` runs this test to make sure the results of a forward pass in frugally-deep are the same as if run in Keras.

In order to convert images to `fdeep::tensor3` the convenience function `tensor3_from_bytes` is provided.


Performance
-----------

Currently frugally-deep is not able to keep up with the speed of TensorFlow and its highly optimized code, i.e. alignment, SIMD, kernel fusion and the matrix multiplication of the [Eigen](http://eigen.tuxfamily.org/) library.

```
Duration of a single forward pass
---------------------------------

| Model       | Keras + TensorFlow | frugally-deep |
|-------------|--------------------|---------------|
| InceptionV3 |             1.10 s |        1.67 s |
| ResNet50    |             0.98 s |        1.18 s |
| VGG16       |             1.32 s |        4.43 s |
| VGG19       |             1.47 s |        5.68 s |
| Xception    |             1.83 s |        2.65 s |

*measured using GCC -O3
 and run on a single core of an Intel Core i5-6600 CPU @ 3.30GHz
 Keras Version 2.0.8, TensorFlow 1.3.0
```

However frugally-deeps offers other beneficial properties like low RAM usage, small library size and ease of use regarding Keras import and integration. GPU usage is not supported.


Requirements and Installation
-----------------------------

A **C++14**-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.7 (libc++ 3.7) and Visual C++ 2015.

You can install frugally-deep using cmake as shown below, or (if you prefer) download the [code](https://github.com/Dobiasd/frugally-deep/archive/master.zip) (and the [code](https://github.com/Dobiasd/FunctionalPlus/archive/master.zip) of [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)), extract it and tell your compiler to use the `include` directories.

```
git clone https://github.com/Dobiasd/FunctionalPlus
cd FunctionalPlus
mkdir build
cd build
cmake ..
make
sudo make install
cd ..
git clone https://github.com/Dobiasd/frugally-deep
cd frugally-deep
mkdir build
cd build
cmake ..
make
sudo make install
cd..
```

Building the tests (optional) requires [doctest](https://github.com/onqtam/doctest). Unit Tests are disabled by default – they are enabled and executed by:

```
cmake -DUNITTEST=ON ..
make unittest
```


Internals
---------

frugally-deep uses `channels_first` (`(depth/channels, height, width`) as its `image_data_format` internally. `export_model.py` takes care of all necessary conversions.
From then on everything is handled as a float32 tensor with rank 3. Dense layers for example take its input flattened to a shape of `(n, 1, 1)`. This is also the shape you will receive as the output of a final `softmax` layer for example.



todo
----

release:
- re-enable logo
- github project description: Use Keras models in C++ with this small header-only library.
- add github project tags
- post on
 - https://www.reddit.com/r/deeplearning/
 - https://www.reddit.com/r/cpp/
 - https://www.reddit.com/r/programming/
 - https://www.reddit.com/r/KerasML/
 - https://www.reddit.com/r/MachineLearning/

add merge layers (https://keras.io/layers/merge/)

support dilated convolution (Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1)

make pure tensorflow example for offset in conv, and sepConv depeding on input_depth and ask on SO

optional low-memory mode (no im2col)

empty cache when possible to save RAM?

use faster GEMM to make im2col worthwhile