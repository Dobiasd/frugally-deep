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

#include "frugally_deep/layers.h"

namespace fd
{

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
layer_ptr bottleneck_sandwich_dims_individual(
    const size3d& size_in,
    const size2d& filter_size,
    const layer_ptr& activation_layer_intermediate,
    const layer_ptr& activation_layer)
{
    assert(activation_layer);
    assert(size_in.depth_ % 2 == 0);
    std::size_t c = size_in.depth_;
    size2d filter_size_x(1, filter_size.width_);
    size2d filter_size_y(filter_size.height_, 1);
    size3d size_intermediate(c / 2, size_in.height_, size_in.width_);
    layer_ptrs layers = {
        conv(size_in, size2d(1, 1), c / 2, 1), activation_layer_intermediate,
        conv(size_intermediate, filter_size_x, c / 2, 1), activation_layer_intermediate,
        conv(size_intermediate, filter_size_y, c / 2, 1), activation_layer_intermediate,
        conv(size_intermediate, size2d(1, 1), c, 1), activation_layer
        };
    return std::make_shared<fd::multi_layer_net>(layers);
}

layer_ptr leaky_relu(const size3d& size_in, float_t alpha)
{
    return std::make_shared<leaky_relu_layer>(size_in, alpha);
}

layer_ptr leaky_relu(std::size_t width_in, float_t alpha)
{
    return std::make_shared<leaky_relu_layer>(size3d(1, 1, width_in), alpha);
}

layer_ptr relu(const size3d& size_in)
{
    return std::make_shared<relu_layer>(size_in);
}

layer_ptr relu(std::size_t width_in)
{
    return std::make_shared<relu_layer>(size3d(1, 1, width_in));
}

layer_ptr tanh(const size3d& size_in)
{
    return std::make_shared<tanh_layer>(size_in);
}

layer_ptr tanh(std::size_t width_in)
{
    return std::make_shared<tanh_layer>(size3d(1, 1, width_in));
}

layer_ptr sigmoid(const size3d& size_in)
{
    return std::make_shared<sigmoid_layer>(size_in);
}

layer_ptr sigmoid(std::size_t width_in)
{
    return std::make_shared<sigmoid_layer>(size3d(1, 1, width_in));
}

layer_ptr max_pool(const size3d& size_in, std::size_t scale_factor)
{
    return std::make_shared<max_pool_layer>(size_in, scale_factor);
}

layer_ptr avg_pool(const size3d& size_in, std::size_t scale_factor)
{
    return std::make_shared<avg_pool_layer>(size_in, scale_factor);
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

layer_ptr softmax(std::size_t width_in)
{
    return std::make_shared<softmax_layer>(size3d(1, 1, width_in));
}

layer_ptr step(const size3d& size_in)
{
    return std::make_shared<step_layer>(size_in);
}

layer_ptr step(std::size_t width_in)
{
    return std::make_shared<step_layer>(size3d(1, 1, width_in));
}

layer_ptr net(const layer_ptrs& layers)
{
    return std::make_shared<multi_layer_net>(layers);
}

} // namespace fd
