/*
https://github.com/Dobiasd/frugally-deep/issues/435
https://keras.io/api/layers/convolution_layers/convolution1d_transpose/
https://www.reddit.com/r/learnmachinelearning/comments/1byw5lb/understanding_conv2dtranspose/
https://arxiv.org/pdf/1603.07285
https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
https://stackoverflow.com/questions/39373230/what-does-tensorflows-conv2d-transpose-operation-do
https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba/
*/

/*
import numpy as np
import keras

x = np.array([[[[10.0, 20.0]]]])

l = keras.layers.Conv2DTranspose(3, 1, use_bias=False)
l.set_weights(np.array([[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]]))

#l = keras.layers.Conv2D(3, 1, use_bias=False)
#l.set_weights(np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]]))

y = l(x)

x
l.weights[0].value
y.numpy()
*/

/*
    const auto w = fdeep::float_vec({1.0, 2.0, 3.0, 4.0});
    const auto l = fdeep::internal::conv_2d_transpose_layer(
        "t",
        fdeep::tensor_shape(1, 1, 2),
        2,
        fdeep::internal::shape2(1, 1),
        fdeep::internal::padding::valid,
        fdeep::internal::shape2(1, 1),
        w,
        fdeep::float_vec({0.0, 0.0})
    );

    const auto x = fdeep::tensor(fdeep::tensor_shape(1, 1, 1, 2), fdeep::float_vec({10.0, 20.0}));
    const auto y = l.apply({x}).front();
    std::cout << fdeep::show_tensor(x) << std::endl;
    std::cout << fplus::show_cont(w) << std::endl;
    std::cout << fdeep::show_tensor(y) << std::endl;
*/

#include "fdeep/fdeep.hpp"

int main()
{
    fdeep::load_model("test_model_exhaustive.json");
}
