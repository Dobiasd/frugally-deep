frugally-deep FAQ
=================

Why is my prediction roughly 100 times slower in C++ than in Python?
------------------------------------------------------------------

Maybe you did not tell your C++ compiler to optimize for speed.
For g++ and clang this can be done with `-O3` (and `-march=native`).
In the case of Microsoft Visual C++,
you need to compile your project not in "Debug" mode but in "Release" mode,
and then run it without the debugger attached.

Why is my prediction roughly 10 times slower in C++ than in Python?
-----------------------------------------------------------------

Maybe you are using your GPU in TensorFlow?
Frugally-deep does not support GPUs.
If you'd like to [compare the performance](test/Dockerfile) of both libraries,
disable the CPU for TensorFlow (`CUDA_VISIBLE_DEVICES=''`).

Why is my prediction roughly 4 times slower in C++ than in Python?
----------------------------------------------------------------

TensorFlow uses multiple CPU cores, even for one prediction, if available.
Frugally-deep does not do that.
If you'd like to [compare the performance](test/Dockerfile) of both libraries,
allow only one CPU core to be used for TensorFlow (`taskset --cpu-list 1`).

If you want more overall throughput, you can parallelize more on the "outside".
See ["Does frugally-deep support multiple CPUs?"](#does-frugally-deep-support-multiple-cpus) for details.

Why is my prediction roughly 2 times slower in C++ than in Python?
----------------------------------------------------------------

With single 2D convolutions, frugally-deep is quite fast,
depending on the dimensions even faster than TensorFlow.
But on some models, TensorFlow applies some fancy runtime optimizations,
like kernel fusion, etc. Frugally-deep does not support such things,
so on some model types, you might experience an insurmountable
performance difference.

Why is my application using more memory than expected?
------------------------------------------------------

In case you're using glibc, which is the default libc on most major distributions like Ubuntu, Debian, Arch, etc., memory temporarily allocated during `fdeep::load_model` [might not be freed completely](https://github.com/nlohmann/json#memory-release). To make sure it's given back to the operating system, use [`malloc_trim(0);`](https://manned.org/malloc_trim.3) after calling `fdeep::load_model`.

Why do I get an error when loading my `.json` file in C++?
------------------------------------------------------------

Most likely it's one of the following two reasons:

- The TensorFlow version used is not the one listed in the [requirements](README.md#requirements-and-installation).
- The conversion from `.h5` to `.json` (using `convert_model.py`) was not done with the same version as used when loading the model in C++.

In case you've made sure none of the above is the cause, please open [an issue](https://github.com/Dobiasd/frugally-deep/issues) with a minimal example to reproduce the problem.

Why does `fdeep::model::predict` take and return multiple `fdeep::tensor`s and not just one tensor?
----------------------------------------------------------------------------------------------------

Only Keras models created with the [sequential API](https://keras.io/getting-started/sequential-model-guide/) must have only one input and output tensor.
Models made with the [functional API](https://keras.io/getting-started/functional-api-guide/) can have multiple inputs and outputs.

This `fdeep::model::predict` takes (and returns) not one `fdeep::tensor` but an `std::vector` of them (`fdeep::tensors`).

Example:

```python
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Add

inputs = [
    Input(shape=(240, 320, 3)),
    Input(shape=(240, 320, 3))
]

outputs = [
    Concatenate()([inputs[0], inputs[1]]),
    Add()([inputs[0], inputs[1]])
]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='nadam')
model.save('multi_input_and_output_model.h5', include_optimizer=False)
```

Now in C++, we would then also provide (and receive) two tensors:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("multi_input_and_output_model.json");
    const auto result = model.predict({
        fdeep::tensor(fdeep::tensor_shape(240, 320, 3), 42),
        fdeep::tensor(fdeep::tensor_shape(240, 320, 3), 43)
        });
    std::cout << fdeep::show_tensors(result) << std::endl;
}
```

Keep in mind, giving multiple `fdeep::tensor`s to `fdeep::model::predict` has nothing to do with batch processing because it is not supported. However you can run multiple single predictions in parallel (see the question "Does frugally-deep support multiple CPUs?"), if you want to do that.

Does frugally-deep support multiple CPUs?
-----------------------------------------

Parallel processing for one single prediction is not supported.

However if you have multiple predictions to make,
you can make use of the fact that a frugally-deep model is thread-safe,
i.e., you can call `model.predict` on the same model instance from different threads simultaneously.
This way you may utilize up to as many CPU cores as you have predictions to make.
In addition, with `model::predict_multi` there is a convenience function available to handle the parallelism for you.
This however is not equivalent to batch processing in Keras,
since each forward pass will still be made in isolation.

How to do regression vs. classification?
----------------------------------------

`fdeep::model::predict` is the generic prediction.

In case you are doing classification,
your model might have a softmax as the last layer.
Then you will get one tensor with a probability for each possible class.
`fdeep::model::predict_class` is a convenience wrapper that will run the forward pass
and return the predicted class number,
so you don't need to manually find the position in the output tensor with the highest activation.

In case you are doing regression resulting in one single value, you can use
`fdeep::model::predict_single_output`,
which will only return one single floating-point value instead of `tensor`s.

Which data format is used internally?
-------------------------------------

frugally-deep uses `channels_last` (`height, width, depth/channels`) as its internal `image_data_format`, as does TensorFlow.
Everything is handled as a float tensor with rank 5.
In case of color images, the first two dimensions of the tensor will have size `1`.

Why does my model return different values with frugally-deep compared to Keras?
-------------------------------------------------------------------------------

The fact that `fdeep::load_model` (with default settings) did not fail,
already proves that your model works the same with frugally-deep as it does with Keras,
because when using `convert_model.py` a test case (input and corresponding output values) is generated automatically and saved along with your model. `fdeep::load_model` runs this test to make sure the results of a forward pass in frugally-deep are the same as in Keras.
If not, an exception is thrown.

So why do you get different values nonetheless when running `fdeep::model::predict`?
Probably you are not feeding the exact same values into the model as you do in Python.
Especially in the case of images as input, this can be caused by:

* different normalization method of the pixel values
* different ways to scale (e.g., interpolation mode) the image before using it

To check if the input values really are the same, you can print them, in Python and in C++:

```python
input = ...
print(input)
print(input.shape)
result = model.predict([input])
print(result)
print(result.shape)  # result[0].shape in case of multiple output tensors
```

```cpp
const fdeep::tensor input = ...
std::cout << fdeep::show_tensor(input);
std::cout << fdeep::show_tensor_shape(input.shape());
const auto result = model.predict({input});
std::cout << fdeep::show_tensor_shape(result.front().shape());
std::cout << fdeep::show_tensors(result);
```

And then check if they actually are identical.

In case you are creating your `fdeep::tensor input` using `fdeep::tensor_from_bytes`,
this way you will also implicitly check if you are using the correct values for `high` and `low` in the call to it.

What to do when loading my model with frugally-deep throws an `std::runtime_error` with `test failed`?
------------------------------------------------------------------------------------------------------

Frugally-deep makes sure your model works exactly the same in C++ as it does in Python by running a test when loading.

You can soften these tests by increasing `verify_epsilon` in the call to `fdeep::load_model`,
or even disable them completely by setting `verify` to `false`.

Also, you might want to try to use `double` instead of `float` for more precision,
which you can do by inserting:

```cpp
#define FDEEP_FLOAT_TYPE double
```

before your first include of `fdeep.hpp`:

```cpp
#include <fdeep/fdeep.hpp>
```

Doing so, however, will increase the memory usage of your application and might slow it down a bit.


How to silence the logging output of `fdeep::model::load`?
----------------------------------------------------------

You can use `fdeep::dev_null_logger` for this:

```cpp
const auto model = fdeep::load_model("model.json", true, fdeep::dev_null_logger);
```

Why does `fdeep::model` not have a default constructor?
-------------------------------------------------------

Because an empty model does not make much sense.
And instead of letting it by convention just forward the input
or raise an exception when `.predict` is invoked,
it can only be constructed by `fdeep::load_model` / `fdeep::read_model`.
This way it is guaranteed you always have a valid model.

In case you would like to, for example, use `fdeep::model` as a member variable of a custom class,
and you want to initialize it not directly during construction of your objects, you can
express this kind of optionality by using `std::unique_ptr<fdeep::model>`
or `fplus::maybe<fdeep::model>`.

How to use images loaded with [CImg](http://cimg.eu/) as input for a model?
---------------------------------------------------------------------------

The following example code shows how to:

* load an image using CImg
* convert it to a `fdeep::tensor`
* use it as input for a forward pass on an image-classification model
* print the class number

```cpp
#include <fdeep/fdeep.hpp>
#include <CImg.h>

fdeep::tensor cimg_to_tensor(const cimg_library::CImg<unsigned char>& image,
    fdeep::float_type low = 0.0f, fdeep::float_type high = 1.0f)
{
    const int width = image.width();
    const int height = image.height();
    const int channels = image.spectrum();

    std::vector<unsigned char> pixels;
    pixels.reserve(height * width * channels);

    // CImg stores the pixels of an image non-interleaved:
    // http://cimg.eu/reference/group__cimg__storage.html
    // This loop changes the order to interleaved,
    // e.e. RRRGGGBBB to RGBRGBRGB for 3-channel images.
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                pixels.push_back(image(x, y, 0, c));
            }
        }
    }

    return fdeep::tensor_from_bytes(pixels.data(), height, width, channels,
        low, high);
}

int main()
{
    const cimg_library::CImg<unsigned char> image("image.jpg");
    const auto model = fdeep::load_model("model.json");
    // Use the correct scaling, i.e., low and high.
    const auto input = cimg_to_tensor(image, 0.0f, 1.0f);
    const auto result = model.predict_class({input});
    std::cout << result << std::endl;
}
```

How to use images loaded with [OpenCV](https://opencv.org/) as input for a model?
---------------------------------------------------------------------------------

The following example code shows how to:

* load an image using OpenCV
* convert it to a `fdeep::tensor`
* use it as input for a forward pass on an image-classification model
* print the class number

```cpp
#include <fdeep/fdeep.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    const cv::Mat image = cv::imread("image.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    assert(image.isContinuous());
    const auto model = fdeep::load_model("model.json");
    // Use the correct scaling, i.e., low and high.
    const auto input = fdeep::tensor_from_bytes(image.ptr(),
        static_cast<std::size_t>(image.rows),
        static_cast<std::size_t>(image.cols),
        static_cast<std::size_t>(image.channels()),
        0.0f, 1.0f);
    const auto result = model.predict_class({input});
    std::cout << result << std::endl;
}
```

How to convert an `fdeep::tensor` to an (OpenCV) image and back?
----------------------------------------------------------------

Example code for how to:

* Convert an OpenCV image to an `fdeep::tensor`
* Convert an `fdeep::tensor` to an OpenCV image

```cpp
#include <fdeep/fdeep.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    const cv::Mat image1 = cv::imread("image.jpg");

    // convert cv::Mat to fdeep::tensor (image1 to tensor)
    const fdeep::tensor tensor =
        fdeep::tensor_from_bytes(image1.ptr(),
            image1.rows, image1.cols, image1.channels());

    // choose the correct pixel type for cv::Mat (gray or RGB/BGR)
    assert(tensor.shape().depth_ == 1 || tensor.shape().depth_ == 3);
    const int mat_type = tensor.shape().depth_ == 1 ? CV_8UC1 : CV_8UC3;
    const int mat_type_float = tensor.shape().depth_ == 1 ? CV_32FC1 : CV_32FC3;

    // convert fdeep::tensor to byte cv::Mat (tensor to image2)
    const cv::Mat image2(
        cv::Size(tensor.shape().width_, tensor.shape().height_), mat_type);
    fdeep::tensor_into_bytes(tensor,
        image2.data, image2.rows * image2.cols * image2.channels());

    // convert fdeep::tensor to float cv::Mat (tensor to image3)
    const cv::Mat image3(
        cv::Size(tensor.shape().width_, tensor.shape().height_), mat_type_float);
    const auto values = tensor.to_vector();
    std::memcpy(image3.data, values.data(), values.size() * sizeof(float));

    // normalize float cv::Mat into float cv::Mat (image3 to image4)
    cv::Mat image4;
    cv::normalize(image3, image4, 1.0, 0.0, cv::NORM_MINMAX);

    // normalize float cv::Mat into byte cv::Mat (image3 to image5)
    cv::Mat tempImage5;
    cv::Mat image5;
    cv::normalize(image3, tempImage5, 255.0, 0.0, cv::NORM_MINMAX);
    tempImage5.convertTo(image5, mat_type);

    // show images for visual verification
    cv::imshow("image1", image1);
    cv::imshow("image2", image2);
    cv::imshow("image3", image3);
    cv::imshow("image4", image4);
    cv::imshow("image5", image5);
    cv::waitKey();
}
```

How to convert an `Eigen::Matrix` to `fdeep::tensor`?
------------------------------------------------------

You can copy the values from `Eigen::Matrix` to `fdeep::tensor`:

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fdeep/fdeep.hpp>

int main()
{
    // dimensions of the eigen matrix
    const int rows = 640;
    const int cols = 480;

    // matrix having its own memory
    Eigen::MatrixXf mat(rows, cols);

    // populate mapped_matrix some way
    mat(0, 0) = 4.0f;
    mat(1, 1) = 5.0f;
    mat(4, 2) = 6.0f;

    // create fdeep::tensor with its own memory
    const int tensor_channels = 1;
    const int tensor_rows = rows;
    const int tensor_cols = cols;
    fdeep::tensor_shape tensor_shape(tensor_rows, tensor_cols, tensor_channels);
    fdeep::tensor t(tensor_shape, 0.0f);

    // copy the values into tensor

    for (int y = 0; y < tensor_rows; ++y)
    {
        for (int x = 0; x < tensor_cols; ++x)
        {
           for (int c = 0; c < tensor_channels; ++c)
            {
                t.set(fdeep::tensor_pos(y, x, c), mat(y, x));
            }
        }
    }

    // print some values to make sure the mapping is correct
    std::cout << t.get(fdeep::tensor_pos(0, 0, 0)) << std::endl;
    std::cout << t.get(fdeep::tensor_pos(1, 1, 0)) << std::endl;
    std::cout << t.get(fdeep::tensor_pos(4, 2, 0)) << std::endl;
}
```

How to fill an `fdeep::tensor` with values, e.g., from an `std::vector<float>`?
--------------------------------------------------------------------------------

Of course one can use `fdeep::tensor` as the primary data structure and fill it with values like so:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    fdeep::tensor t(fdeep::tensor_shape(3, 1, 1), 0);
    t.set(fdeep::tensor_pos(0, 0, 0), 1);
    t.set(fdeep::tensor_pos(1, 0, 0), 2);
    t.set(fdeep::tensor_pos(2, 0, 0), 3);
}
```

In case one already has an `std::vector<float>` with values, one can just construct a `fdeep::tensor` from it, holding a copy of the values:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const std::vector<float> v = {1, 2, 3};
    const fdeep::tensor t(fdeep::tensor_shape(3, 1, 1), v);
}
```

How to convert an `fdeep::tensor` to an `std::vector<float>`?
--------------------------------------------------------------

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const fdeep::tensor tensor(
        fdeep::tensor_shape(static_cast<std::size_t>(4)),
        std::vector<float>{1, 2, 3, 4});
    const std::vector<float> vec = tensor.to_vector();
}
```

Why are `Conv2DTranspose` layers not supported?
-----------------------------------------------

The combination of `UpSampling2D` and `Conv2D` layers seems to be the better alternative:
https://distill.pub/2016/deconv-checkerboard/

Basically, instead of this:

```python
x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(x)
```

one uses that:

```python
x = Conv2D(8, (3, 3), padding='same')(UpSampling2D(2)(x))
```

In case you are not in the position to change your model's
architecture to make that change,
feel free to implement `Conv2DTranspose` in frugally-deep and
submit a [pull request](https://github.com/Dobiasd/frugally-deep/pulls). :)

How can I use `BatchNormalization` and `Dropout` layers with `training=True`?
-----------------------------------------------------------------------------

Frugally-deep does not support `training=True` on the inbound nodes.

But if you'd like to remove this flag from the layers in your model,
you can use the following function to do so before using `convert_model.py`:

```python3
def remove_training_flags(old_model_path, new_model_path):
    def do_remove(model):
        layers = model.layers
        for layer in layers:
            for node in layer.inbound_nodes:
                if "training" in node.call_kwargs and node.call_kwargs["training"] is True:
                    print(f"Removing training=True from inbound node to layer named {layer.name}.")
                    del node.call_kwargs["training"]
            layer_type = type(layer).__name__
            if layer_type in ['Model', 'Sequential', 'Functional']:
                do_remove(layer)
        return model

    do_remove(load_model(old_model_path)).save(new_model_path, include_optimizer=False)
```

Why are `Lambda` layers not supported?
-----------------------------------------------

`Lambda` layers in Keras involve custom Python code to be executed.
Supporting this in frugally-deep would require having a transpiler from Python to C++,
aware of the semantic differences between the data structures too.
Since this is not feasible, `Lambda` layers are not supported in frugally-deep.

In case you don't find a way to get rid of the `Lambda` layer in your Keras model,
feel free to dive into the rabbit hole of [injecting support for your custom layers to frugally-deep](FAQ.md#how-to-use-custom-layers).

How to use custom layers?
-------------------------

`fdeep::load_model` has a `custom_layer_creators` parameter,
which is of the following type:
```cpp
const std::unordered_map<
    std::string,
    std::function<layer_ptr(
        const get_param_f&,
        const get_global_param_f&,
        const nlohmann::json&,
        const std::string&)>>&
```

It is a dictionary, mapping layer names to custom factory functions.
As an example for such a factory function for a simple layer type,
please have a look at the definition of `fdeep::internal::create_add_layer`.

So, you provide your own factory function,
returning an `fdeep::internal::layer_ptr` (`std::shared_ptr<fdeep::layer>`).

For the actual implementation of your layer, you need to create a new class,
inheriting from `fdeep::internal::layer`.
As an example, please have a look at the definition of the `add_layer` class.

In summary, the work needed to inject support for a custom layer
from userland, i.e., without modifying the actual library,
looks as follows:
- Create a new layer class, inheriting from `fdeep::layer`, like [so](https://github.com/Dobiasd/frugally-deep/blob/e3e1a6a2e011ef6255d6589a5ec0981c9d0ef1f9/include/fdeep/layers/add_layer.hpp#L16).
- Create a new creator function for your layer type, like [so](https://github.com/Dobiasd/frugally-deep/blob/e3e1a6a2e011ef6255d6589a5ec0981c9d0ef1f9/include/fdeep/import_model.hpp#L594)
- Pass a `custom_layer_creators` to `fdeep::load_model`, which maps layer names to your custom creators.

In case your layer is trainable, i.e., you have some weights attached to it,
that also need to be exported from the Python side of things,
have a look into `convert_model.py` for how to extend the resulting
model `.json` file with the parameters you need.

Remark: This feature in general is still experimental and might be subject to
change in the future, as the usage of namespace `fdeep::internal` indicates.
