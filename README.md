![logo](logo/fdeep.png)

[![Build Status](https://travis-ci.org/Dobiasd/frugally-deep.svg?branch=master)][travis]
[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[travis]: https://travis-ci.org/Dobiasd/frugally-deep
[license]: LICENSE


frugally-deep
=============
**Use Keras models in C++ with ease**


Table of contents
-----------------
  * [Introduction](#introduction)
  * [Usage](#usage)
  * [Performance](#performance)
  * [Requirements and Installation](#requirements-and-installation)
  * [Internals](#internals)


Introduction
------------

Would you like to build/train a model using Keras/Python? And would you like run the prediction (forward pass) on your model in C++ without linking your application against TensorFlow? Then frugally-deep is exactly for you.

**frugally-deep**

* **is a small header-only library** written in modern and pure C++.
* is very easy to integrate and use.
* depends only on [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus), [Eigen](http://eigen.tuxfamily.org/) and [json](https://github.com/nlohmann/json) - also header-only libraries.
* supports inference (`model.predict`) not only for [sequential models](https://keras.io/getting-started/sequential-model-guide/) but also for computational graphs with a more complex topology, created with the [functional API](https://keras.io/getting-started/functional-api-guide/).
* re-implements a (small) subset of TensorFlow, i.e. the operations needed to support prediction.
* results in a much smaller binary size than linking against TensorFlow.
* works out of-the-box also when compiled into a 32-bit executable.
* utterly ignores even the most powerful GPU in your system and uses only one CPU core. ;-)
* but is quite fast on one CPU core [compared to TensorFlow](#performance).


### Supported layer types

Layer types typically used in image recognition/generation are supported, making many popular model architectures possible (see [Performance section](#performance)).

* `Add`, `Concatenate`
* `AveragePooling1D/2D`, `GlobalAveragePooling1D/2D`
* `Conv1D/2D`, `SeparableConv2D`
* `Cropping1D/2D`, `ZeroPadding1D/2D`
* `BatchNormalization`, `Dense`, `Dropout`, `Flatten`
* `MaxPooling1D/2D`, `GlobalMaxPooling1D/2D`
* `ELU`, `LeakyReLU`, `ReLU`, `SeLU`
* `Sigmoid`, `Softmax`, `Softplus`, `Tanh`
* `UpSampling1D/2D`


### Also supported

* multiple inputs and outputs
* nested models
* residual connections
* shared layers
* arbitrary complex model architectures / computational graphs


Currently not supported are the following layer types:
`ActivityRegularization`,
`AlphaDropout`,
`Average`,
`AveragePooling3D`,
`Bidirectional`,
`Conv2DTranspose`,
`Conv3D`,
`ConvLSTM2D`,
`CuDNNGRU`,
`CuDNNLSTM`,
`Cropping3D`,
`DepthwiseConv2D`,
`Dot`,
`Embedding`,
`GaussianDropout`,
`GaussianNoise`,
`GRU`,
`GRUCell`,
`Lambda`,
`LocallyConnected1D`,
`LocallyConnected2D`,
`LSTM`,
`LSTMCell`,
`Masking`,
`Maximum`,
`MaxPooling3D`,
`Multiply`,
`Permute`,
`PReLU`,
`RepeatVector`,
`Reshape`,
`RNN`,
`SimpleRNN`,
`SimpleRNNCell`,
`StackedRNNCells`,
`Subtract`,
`ThresholdedReLU`,
`TimeDistributed`,
`Upsampling3D`,
`any custom layers`


Usage
-----

1) Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save('....h5')`. The `image_data_format` in your model must be `channels_last`, which is the default when using the TensorFlow backend. Models created with a different `image_data_format` and other backends are not supported.

2) Now convert it to the frugally-deep file format with `keras_export/convert_model.py`

3) Finally load it in C++ (`fdeep::load_model(...)`) and use `model.predict(...)` to invoke a forward pass with your data.

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
    np.asarray([[1,2,3,4], [2,3,4,5]]),
    np.asarray([[1,0,0], [0,0,1]]), epochs=10)

model.save('keras_model.h5')
```

```
python3 keras_export/convert_model.py keras_model.h5 fdeep_model.json
```

```cpp
// main.cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor3(fdeep::shape3(4, 1, 1), {1, 2, 3, 4})});
    std::cout << fdeep::show_tensor3s(result) << std::endl;
}
```

When using `convert_model.py` a test case (input and corresponding output values) is generated automatically and saved along with your model. `fdeep::load_model` runs this test to make sure the results of a forward pass in frugally-deep are the same as in Keras.

In order to convert images to `fdeep::tensor3` the convenience function `tensor3_from_bytes` is provided.


Performance
-----------

Below you can find the durations of one isolated forward pass for some popular models ran on a single core of an Intel Core i5-6600 CPU @ 3.30GHz. frugally-deep was compiled with `g++ -O3`. The processes were started with `CUDA_VISIBLE_DEVICES='' taskset --cpu-list 1 ...` to disable the GPU and to only allow usage of one CPU.

```
| Model             | Keras + TensorFlow | frugally-deep |
|-------------------|--------------------|---------------|
| InceptionV3       |             1.21 s |        0.54 s |
| ResNet50          |             0.95 s |        0.34 s |
| VGG16             |             1.15 s |        1.37 s |
| VGG19             |             1.45 s |        1.67 s |
| Xception          |             1.63 s |        0.85 s |
| DenseNet201       |             2.49 s |        0.49 s |
| NASNetLarge       |             5.86 s |        3.12 s |
```

versions: `Keras 2.1.3`, `TensorFlow 1.5.0` (default packages from pip)

Using `-march=native` when compiling frugally-deep brings the times further down but these values would have to be compared with the ones resulting from a TensorFlow version with the same optimizations.


Requirements and Installation
-----------------------------

A **C++14**-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.7 (libc++ 3.7) and Visual C++ 2015.

You can install frugally-deep using cmake as shown below, or (if you prefer) download the [code](https://github.com/Dobiasd/frugally-deep/archive/master.zip) (and the [code](https://github.com/Dobiasd/FunctionalPlus/archive/master.zip) of [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)), extract it and tell your compiler to use the `include` directories.

```
git clone https://github.com/Dobiasd/FunctionalPlus
cd FunctionalPlus
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

sudo apt install mercurial
hg clone https://bitbucket.org/eigen/eigen/
cd eigen
mkdir -p build && cd build
cmake ..
make && sudo make install
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
cd ../..

git clone https://github.com/nlohmann/json
cd json
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

git clone https://github.com/Dobiasd/frugally-deep
cd frugally-deep
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..
```

Building the tests (optional) requires [doctest](https://github.com/onqtam/doctest). Unit Tests are disabled by default – they are enabled and executed by:

```
cmake -DFDEEP_UNITTEST=ON ..
make unittest
```


Internals
---------

frugally-deep uses `channels_first` (`depth/channels, height, width`) as its `image_data_format` internally. `convert_model.py` takes care of all necessary conversions.
From then on everything is handled as a float32 tensor with rank 3. Dense layers for example take its input flattened to a shape of `(n, 1, 1)`. This is also the shape you will receive as the output of a final `softmax` layer for example.

In case you would like to use `double` instead of `float` for all calculations, simply do this:
```cpp
#define FDEEP_FLOAT_TYPE double
#include <fdeep/fdeep.hpp>
```

A frugally-deep model is thread-safe, i.e. you can call `model.predict` on the same model instance from different threads simultaneously. This way you may utilize up to as many CPU cores as you have predictions to make. With `model::predict_multi` there is a convenience function available to handle the parallelism for you.


Disclaimer
----------
The API of this library still might change in the future. If you have any suggestions, find errors or want to give general feedback/criticism, I'd [love to hear from you](https://github.com/Dobiasd/frugally-deep/issues). Of course, [contributions](https://github.com/Dobiasd/frugally-deep/pulls) are also very welcome.


License
-------
Distributed under the MIT License.
(See accompanying file [`LICENSE`](https://github.com/Dobiasd/frugally-deep/blob/master/LICENSE) or at
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))
