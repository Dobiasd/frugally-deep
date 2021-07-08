![logo](logo/fdeep.png)

[![CI](https://github.com/Dobiasd/frugally-deep/workflows/ci/badge.svg)](https://github.com/Dobiasd/frugally-deep/actions)
[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

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
* [FAQ](#faq)

Introduction
------------

Would you like to build/train a model using Keras/Python? And would you like to run the prediction (forward pass) on your model in C++ without linking your application against TensorFlow? Then frugally-deep is exactly for you.

**frugally-deep**

* **is a small header-only library** written in modern and pure C++.
* is very easy to integrate and use.
* depends only on [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus), [Eigen](http://eigen.tuxfamily.org/) and [json](https://github.com/nlohmann/json) - also header-only libraries.
* supports inference (`model.predict`) not only for [sequential models](https://keras.io/getting-started/sequential-model-guide/) but also for computational graphs with a more complex topology, created with the [functional API](https://keras.io/getting-started/functional-api-guide/).
* re-implements a (small) subset of TensorFlow, i.e., the operations needed to support prediction.
* results in a much smaller binary size than linking against TensorFlow.
* works out-of-the-box also when compiled into a 32-bit executable. (Of course, 64 bit is fine too.)
* utterly ignores even the most powerful GPU in your system and uses only one CPU core per prediction. ;-)
* but is quite fast on one CPU core [compared to TensorFlow](#performance), and you can run multiple predictions in parallel, thus utilizing as many CPUs as you like to improve the overall prediction throughput of your application/pipeline.


### Supported layer types

Layer types typically used in image recognition/generation are supported, making many popular model architectures possible (see [Performance section](#performance)).

* `Add`, `Concatenate`, `Subtract`, `Multiply`, `Average`, `Maximum`
* `AveragePooling1D/2D`, `GlobalAveragePooling1D/2D`
* `Bidirectional`, `TimeDistributed`, `GRU`, `LSTM`, `CuDNNGRU`, `CuDNNLSTM`
* `Conv1D/2D`, `SeparableConv2D`, `DepthwiseConv2D`
* `Cropping1D/2D`, `ZeroPadding1D/2D`
* `BatchNormalization`, `Dense`, `Flatten`
* `Dropout`, `AlphaDropout`, `GaussianDropout`, `GaussianNoise`, `RandomRotation`
* `SpatialDropout1D`, `SpatialDropout2D`, `SpatialDropout3D`
* `MaxPooling1D/2D`, `GlobalMaxPooling1D/2D`
* `ELU`, `LeakyReLU`, `ReLU`, `SeLU`, `PReLU`
* `Sigmoid`, `Softmax`, `Softplus`, `Tanh`
* `UpSampling1D/2D`
* `Reshape`, `Permute`
* `Embedding`


### Also supported

* multiple inputs and outputs
* nested models
* residual connections
* shared layers
* variable input shapes
* arbitrary complex model architectures / computational graphs
* custom layers (by passing custom factory functions to `load_model`)

### Currently not supported are the following:

`ActivityRegularization`,
`AveragePooling3D`,
`Conv2DTranspose` ([why](FAQ.md#why-are-conv2dtranspose-layers-not-supported)),
`Conv3D`,
`ConvLSTM2D`,
`Cropping3D`,
`Dot`,
`GRUCell`,
`LocallyConnected1D`,
`LocallyConnected2D`,
`LSTMCell`,
`Masking`,
`MaxPooling3D`,
`RepeatVector`,
`RNN`,
`SimpleRNN`,
`SimpleRNNCell`,
`StackedRNNCells`,
`ThresholdedReLU`,
`Upsampling3D`,
`temporal` models

Usage
-----

1) Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save('....h5', include_optimizer=False)`. The `image_data_format` in your model must be `channels_last`, which is the default when using the TensorFlow backend. Models created with a different `image_data_format` and other backends are not supported.

2) Now convert it to the frugally-deep file format with `keras_export/convert_model.py`

3) Finally load it in C++ (`fdeep::load_model(...)`) and use `model.predict(...)` to invoke a forward pass with your data.

The following minimal example shows the full workflow:

```python
# create_model.py
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(4,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]]),
    np.asarray([[1, 0, 0], [0, 0, 1]]), epochs=10)

model.save('keras_model.h5', include_optimizer=False)
```

```bash
python3 keras_export/convert_model.py keras_model.h5 fdeep_model.json
```

```cpp
// main.cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),
        std::vector<float>{1, 2, 3, 4})});
    std::cout << fdeep::show_tensors(result) << std::endl;
}
```

When using `convert_model.py` a test case (input and corresponding output values) is generated automatically and saved along with your model. `fdeep::load_model` runs this test to make sure the results of a forward pass in frugally-deep are the same as in Keras.

For more integration examples please have a look at the [FAQ](FAQ.md).

Performance
-----------

Below you can find the average durations of multiple consecutive forward passes for some popular models ran on a **single core** of an Intel Core i5-6600 CPU @ 3.30GHz. frugally-deep and TensorFlow were compiled (GCC ver. 7.1) with `g++ -O3 -march=native`. The processes were started with `CUDA_VISIBLE_DEVICES='' taskset --cpu-list 1 ...` to **disable the GPU** and to only allow usage of one CPU.
(see used [`Dockerfile`](test/Dockerfile))

| Model             | Keras + TF | frugally-deep |
| ----------------- | ----------:| -------------:|
| `DenseNet121`     |     0.12 s |        0.25 s |
| `DenseNet169`     |     0.13 s |        0.28 s |
| `DenseNet201`     |     0.16 s |        0.39 s |
| `InceptionV3`     |     0.21 s |        0.32 s |
| `MobileNet`       |     0.05 s |        0.15 s |
| `MobileNetV2`     |     0.05 s |        0.17 s |
| `NASNetLarge`     |     0.83 s |        4.03 s |
| `NASNetMobile`    |     0.08 s |        0.32 s |
| `ResNet101`       |     0.22 s |        0.45 s |
| `ResNet101V2`     |     0.21 s |        0.42 s |
| `ResNet152`       |     0.31 s |        0.65 s |
| `ResNet152V2`     |     0.29 s |        0.61 s |
| `ResNet50`        |     0.13 s |        0.26 s |
| `ResNet50V2`      |     0.12 s |        0.22 s |
| `VGG16`           |     0.40 s |        0.56 s |
| `VGG19`           |     0.49 s |        0.68 s |
| `Xception`        |     0.25 s |        1.20 s |

Requirements and Installation
-----------------------------

- A **C++14**-compatible compiler: Compilers from these versions on are fine: GCC 4.9, Clang 3.7 (libc++ 3.7) and Visual C++ 2015
- Python 3.7 or higher
- TensorFlow 2.5.0

Guides for different ways to install frugally-deep can be found in [`INSTALL.md`](INSTALL.md).

FAQ
---

See [`FAQ.md`](FAQ.md)

Disclaimer

----------
The API of this library still might change in the future. If you have any suggestions, find errors or want to give general feedback/criticism, I'd [love to hear from you](issues). Of course, [contributions](pulls) are also very welcome.

License
-------

Distributed under the MIT License.
(See accompanying file [`LICENSE`](LICENSE) or at
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))
