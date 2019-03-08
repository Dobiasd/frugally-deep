frugally-deep FAQ
=================

Why is my prediction roughly 100 times slower in C++ as in Python?
------------------------------------------------------------------

Maybe you did not tell your C++ compiler to optimize for speed?
For g++ and clang this can be done with `-O3`.
In case of Microsoft Visual C++,
you need to compile your project not in "Debug" mode but in "Release" mode,
and then run it without the debugger attached.

Why does `fdeep::model::predict` take and return multiple `fdeep::tensor5`s and not just one tensor?
----------------------------------------------------------------------------------------------------

Only Keras models created with the [sequential API](https://keras.io/getting-started/sequential-model-guide/) must have only one input tensor and output tensor.
Models make with the [functional API](https://keras.io/getting-started/functional-api-guide/) can have multiple inputs and outputs.

This `fdeep::model::predict` takes (and returns) not one `fdeep::tensor5` but an `std::vector` of them (`fdeep::tensor5s`).

Example:

```python
from keras.models import Model
from keras.layers import Input, Concatenate, Add

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
        fdeep::tensor5(fdeep::shape5(1, 1, 240, 320, 3), 42),
        fdeep::tensor5(fdeep::shape5(1, 1, 240, 320, 3), 43)
        });
    std::cout << fdeep::show_tensor5s(result) << std::endl;
}
```

Keep in mind, giving multiple `fdeep::tensor5`s to `fdeep::model::predict` this has nothing to do with batch processing, because it is not supported. However you can run multiple single predictions im parallel (see question "Does frugally-deep support multiple CPUs?"), if you want do to that.

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
`fdeep::model::predict` is a convenience wrapper that will run the forward pass
and return the predicted class number,
so you don't need to manually find the position in the output tensor with the highest activation.

In case you are doing regression resulting in one single value, you can use
`fdeep::model::predict_single_output`,
which will only return one single floating-point value instead of `tensor5`s.

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

So why do you different values nonetheless when running `fdeep::model::predict`?
Probably you are not feeding the exact same values into the model as you do in Python.
Especially in the case of images as input this can be caused by:

* different normalization method of the pixel values
* different ways to scale (e.g., interpolation mode) the image before using it

To check if the input values really are the same, you can just print them, in Python and in C++:

```python
input = ...
print(input)
print(input.shape)
result = model.predict([input])
print(result)
print(result.shape)  # result[0].shape in case of multiple output tensors
```

```cpp
const fdeep::tensor5 input = ...
std::cout << fdeep::show_tensor5(input);
std::cout << fdeep::show_shape5(input.shape());
const auto result = model.predict({input});
std::cout << fdeep::show_shape5(result.front().shape());
std::cout << fdeep::show_tensor5s(result);
```

And then check if they actually are identical.

In case you are creating your `fdeep::tensor5 input` using `fdeep::tensor5_from_bytes`,
this way you will also implicitly check if you are using the correct values for `high` and `low` in the call to it.

What to do when loading my model with frugally-deep throws an `std::runtime_error` with `test failed`?
------------------------------------------------------------------------------------------------------

Frugally-deep makes sure your model works exactly the same in C++ as it does in Python by running test when loading.

You can soften these tests by increasing `verify_epsilon` in the call to `fdeep::load_model`,
or event disable them completely by setting `verify` to `false`.

Also you might want to try to use `double` instead of `float` for more precision,
which you can do by inserting:

```cpp
#define FDEEP_FLOAT_TYPE double
```

before your first include of `fdeep.hpp`:

```cpp
#include <fdeep/fdeep.hpp>
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

The following example code shows for how to:

* load an image using CImg
* convert it to a `fdeep::tensor5`
* use it as input for a forward pass on an image-classification model
* print the class number

```cpp
#include <fdeep/fdeep.hpp>
#include <CImg.h>

fdeep::tensor5 cimg_to_tensor5(const cimg_library::CImg<unsigned char>& image,
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

    return fdeep::tensor5_from_bytes(pixels.data(), height, width, channels,
        low, high);
}

int main()
{
    const cimg_library::CImg<unsigned char> image("image.jpg");
    const auto model = fdeep::load_model("model.json");
    // Use the correct scaling, i.e., low and high.
    const auto input = cimg_to_tensor5(image, 0.0f, 1.0f);
    const auto result = model.predict_class({input});
    std::cout << result << std::endl;
}
```

How to use images loaded with [OpenCV](https://opencv.org/) as input for a model?
---------------------------------------------------------------------------------

The following example code shows for how to:

* load an image using OpenCV
* convert it to a `fdeep::tensor5`
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
    const auto input = fdeep::tensor5_from_bytes(image.ptr(),
        static_cast<std::size_t>(image.rows),
        static_cast<std::size_t>(image.cols),
        static_cast<std::size_t>(image.channels()),
        0.0f, 1.0f);
    const auto result = model.predict_class({input});
    std::cout << result << std::endl;
}
```

How to convert an `fdeep::tensor5` to an (OpenCV) image?
--------------------------------------------------------

Example code for how to:

* convert an OpenCV image to an `fdeep::tensor5`
* convert an `fdeep::tensor5` to an OpenCV image

```cpp
#include <fdeep/fdeep.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    const cv::Mat image1 = cv::imread("image.jpg");

    // convert cv::Mat to fdeep::tensor5 (image1 to tensor)
    const fdeep::tensor5 tensor =
        fdeep::tensor5_from_bytes(image1.ptr(),
            image1.rows, image1.cols, image1.channels());

    // choose the correct pixel type for cv::Mat (gray or RGB/BGR)
    assert(tensor.shape().depth_ == 1 || tensor.shape().depth_ == 3);
    const int mat_type = tensor.shape().depth_ == 1 ? CV_8UC1 : CV_8UC3;

    // convert fdeep::tensor5 to cv::Mat (tensor to image2)
    const cv::Mat image2(
        cv::Size(tensor.shape().width_, tensor.shape().height_), mat_type);
    fdeep::tensor5_into_bytes(tensor,
        image2.data, image2.rows * image2.cols * image2.channels());

    // show both images for visual verification
    cv::imshow("image1", image1);
    cv::imshow("image2", image2);
    cv::waitKey();
}
```

How to convert an `Eigen::Matrix` to `fdeep::tensor5`?
------------------------------------------------------

You can copy the values from `Eigen::Matrix` to `fdeep::tensor5`:

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

    // create fdeep::tensor5 with its own memory
    const int tensor5_channels = 1;
    const int tensor5_rows = rows;
    const int tensor5_cols = cols;
    fdeep::shape5 tensor5_shape(1, 1, tensor5_rows, tensor5_cols, tensor5_channels);
    fdeep::tensor5 t(tensor5_shape, 0.0f);

    // copy the values into tensor5

    for (int y = 0; y < tensor5_rows; ++y)
    {
        for (int x = 0; x < tensor5_cols; ++x)
        {
           for (int c = 0; c < tensor5_channels; ++c)
            {
                t.set(0, 0, y, x, c, mat(y, x));
            }
        }
    }

    // print some values to make sure the mapping is correct
    std::cout << t.get(0, 0, 0, 0, 0) << std::endl;
    std::cout << t.get(0, 0, 0, 1, 1) << std::endl;
    std::cout << t.get(0, 0, 0, 4, 2) << std::endl;
}
```

Or you can reuse the memory, by sharing it between `Eigen::Matrix` and `fdeep::tensor5`.

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fdeep/fdeep.hpp>

int main()
{
    // use row major storage order for eigen matrix, since fdeep uses it too
    using RowMajorMatrixXf = Eigen::Matrix<fdeep::float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // dimensions of the eigen matrix
    const int rows = 640;
    const int cols = 480;

    // initialize memory shared between matrix and tensor
    fdeep::shared_float_vec data_vec = fplus::make_shared_ref<fdeep::float_vec>();
    data_vec->resize(static_cast<std::size_t>(rows * cols));

    // create eigen matrix using the memory block from the vector above
    Eigen::Map<RowMajorMatrixXf, Eigen::Unaligned> mapped_matrix(
        data_vec->data(),
        rows, cols);

    // populate mapped_matrix some way
    mapped_matrix(0, 0) = 4.0f;
    mapped_matrix(1, 1) = 5.0f;
    mapped_matrix(4, 2) = 6.0f;

    // create fdeep::tensor5 also using the memory block of the vector
    const int tensor5_channels = rows;
    const int tensor5_rows = 1;
    const int tensor5_cols = cols;
    fdeep::shape5 tensor5_shape(1, 1, tensor5_rows, tensor5_cols, tensor5_channels);
    fdeep::tensor5 t(tensor5_shape, data_vec);

    // print some values to make sure the mapping is correct
    std::cout << t.get(0, 0, 0, 0, 0) << std::endl;
    std::cout << t.get(0, 0, 0, 1, 1) << std::endl;
    std::cout << t.get(0, 0, 0, 4, 2) << std::endl;
}
```

How to fill an `fdeep::tensor5` with values, e.g., from an `std::vector<float>`?
--------------------------------------------------------------------------------

Of course one can use `fdeep::tensor5` as the primary data structure and fill it with values like so:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    fdeep::tensor5 t(fdeep::shape5(1, 1, 3, 1, 1), 0);
    t.set(0, 0, 0, 0, 0, 1);
    t.set(0, 0, 1, 0, 0, 2);
    t.set(0, 0, 2, 0, 0, 3);
}
```

But in case one already has an `std::vector<float>` with values, one might want to re-use it.

So the `std::vector<float>` needs to be converted to `fplus::shared_ref<std::vector<float>>` for `fdeep::tensor` to accept it in its constructor.

`T` can be converted to `fplus::shared_ref<T>` by using `fplus::make_shared_ref<T>`:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const std::vector<float> v = {1, 2, 3};
    const fdeep::shared_float_vec sv(fplus::make_shared_ref<fdeep::float_vec>(v));
    fdeep::tensor5 t(fdeep::shape5(1, 1, 3, 1, 1), sv);
}
```

In case the original vector is no longer needed, the copy of the value can be avoided by making it an r-value with `std::move`:

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const std::vector<float> v = {1, 2, 3};
    const fdeep::shared_float_vec sv(fplus::make_shared_ref<fdeep::float_vec>(std::move(v)));
    fdeep::tensor5 t(fdeep::shape5(1, 1, 3, 1, 1), sv);
}
```

How to convert an `fdeep::tensor5` to an `std::vector<float>`?
--------------------------------------------------------------

```cpp
#include <fdeep/fdeep.hpp>
int main()
{
    const fdeep::tensor5 tensor(fdeep::shape5(1, 1, 1, 1, 4), {1, 2, 3, 4});
    const std::vector<float> vec = *tensor.as_vector();
}
```
