// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/multi_layer_net.h"

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
layer_ptr conv(const size3d& size_in, const size2d& filter_size,
    std::size_t k, std::size_t stride)
{
    return std::make_shared<convolutional_layer>(size_in, filter_size, k, stride);
}

// conv 1*1, c/2 filters
// conv 1*x, c/2 filters
// conv y*1, c/2 filters
// conv 1*1, c filters
layer_ptr bottleneck_sandwich_3x3_dims_individual(
    const size3d& size_in,
    const size2d& filter_size,
    const layer_ptr& actication_layer_intermediate,
    const layer_ptr& actication_layer)
{
    assert(actication_layer);
    assert(size_in.depth() % 2 == 0);
    std::size_t c = size_in.depth();
    size2d filter_size_x(1, filter_size.width());
    size2d filter_size_y(filter_size.height(), 1);
    size3d size_intermediate(c / 2, size_in.height(), size_in.width());
    layer_ptrs layers = {
        conv(size_in, size2d(1, 1), c / 2, 1), actication_layer_intermediate,
        conv(size_intermediate, filter_size_x, c / 2, 1), actication_layer_intermediate,
        conv(size_intermediate, filter_size_y, c / 2, 1), actication_layer_intermediate,
        conv(size_intermediate, size2d(1, 1), c, 1), actication_layer
        };
    return std::make_shared<fd::multi_layer_net>(layers);
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
