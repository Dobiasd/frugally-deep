// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/layers/avg_pool_layer.h"
#include "frugally_deep/layers/convolutional_layer.h"
#include "frugally_deep/layers/elu_layer.h"
#include "frugally_deep/layers/erf_layer.h"
#include "frugally_deep/layers/fast_sigmoid_layer.h"
#include "frugally_deep/layers/flatten_layer.h"
#include "frugally_deep/layers/fully_connected_layer.h"
#include "frugally_deep/layers/layer.h"
#include "frugally_deep/layers/leaky_relu_layer.h"
#include "frugally_deep/layers/max_pool_layer.h"
#include "frugally_deep/layers/pool_layer.h"
#include "frugally_deep/layers/relu_layer.h"
#include "frugally_deep/layers/sigmoid_layer.h"
#include "frugally_deep/layers/softmax_layer.h"
#include "frugally_deep/layers/softplus_layer.h"
#include "frugally_deep/layers/sigmoid_layer.h"
#include "frugally_deep/layers/tanh_layer.h"
#include "frugally_deep/layers/unpool_layer.h"

namespace fd
{

struct input_with_output
{
    matrix3d input_;
    matrix3d output_;
};
typedef std::vector<input_with_output> input_with_output_vec;

struct classification_dataset
{
    input_with_output_vec training_data_;
    input_with_output_vec test_data_;
};

// todo steps, kann man naemlich statt pool benutzen
layer_ptr conv(const size3d& size_in, const std::size_t& f,
    std::size_t k, std::size_t stride)
{
    return std::make_shared<convolutional_layer>(size_in, f, k, stride);
}

layer_ptr leaky_relu(const size3d& size_in, float_t alpha)
{
    return std::make_shared<leaky_relu_layer>(size_in, alpha);
}

layer_ptr max_pool(const size3d& size_in, std::size_t scale_factor)
{
    return std::make_shared<max_pool_layer>(size_in, scale_factor);
}

layer_ptr flatten(const size3d& size_in)
{
    return std::make_shared<flatten_layer>(size_in);
}

layer_ptr fc(std::size_t n_in, std::size_t n_out)
{
    return std::make_shared<fully_connected_layer>(n_in, n_out);
}

layer_ptr softmax(const size3d& size_in)
{
    return std::make_shared<softmax_layer>(size_in);
}

} // namespace fd
