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

typedef std::function<layer_ptr(const size3d& size_in)> pre_layer;
typedef std::vector<pre_layer> pre_layers;

inline pre_layer conv(const size2d& filter_size,
    std::size_t k, std::size_t stride)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<convolutional_layer>(size_in, filter_size, k, stride);
    };
}

inline pre_layer conv_transp(const size2d& filter_size,
    std::size_t k, std::size_t stride)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<convolution_transpose_layer>(size_in, filter_size, k, stride);
    };
}

inline pre_layer identity()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<identity_layer>(size_in);
    };
}

inline pre_layer leaky_relu(float_t alpha)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<leaky_relu_layer>(size_in, alpha);
    };
}

inline pre_layer elu(float_t alpha)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<elu_layer>(size_in, alpha);
    };
}

inline pre_layer erf()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<erf_layer>(size_in);
    };
}

inline pre_layer relu()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<relu_layer>(size_in);
    };
}

inline pre_layer softplus()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<softplus_layer>(size_in);
    };
}

inline pre_layer tanh(bool snd_deriv_max_at_1 = false, float_t alpha = 0)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<tanh_layer>(size_in, snd_deriv_max_at_1, alpha);
    };
}

inline pre_layer sigmoid()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<sigmoid_layer>(size_in);
    };
}

inline pre_layer fast_sigmoid()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<fast_sigmoid_layer>(size_in);
    };
}

inline pre_layer softmax()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<softmax_layer>(size_in);
    };
}

inline pre_layer step()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<step_layer>(size_in);
    };
}

inline pre_layer max_pool(std::size_t scale_factor)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<max_pool_layer>(size_in, scale_factor);
    };
}

inline pre_layer gentle_max_pool(std::size_t scale_factor, float_t alpha)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<gentle_max_pool_layer>(
            size_in, scale_factor, alpha);
    };
}

inline pre_layer avg_pool(std::size_t scale_factor)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<avg_pool_layer>(size_in, scale_factor);
    };
}

inline pre_layer unpool(std::size_t scale_factor)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<unpool_layer>(size_in, scale_factor);
    };
}

inline pre_layer flatten()
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        return std::make_shared<flatten_layer>(size_in);
    };
}

inline pre_layer fc(std::size_t n_out)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        assert(size_in.depth_ == 1);
        assert(size_in.width_ == 1);
        return std::make_shared<fully_connected_layer>(size_in.height_, n_out);
    };
}

inline pre_layer net(const pre_layers& layer_generators)
{
    return [=](const size3d& size_in) -> layer_ptr
    {
        layer_ptrs layers;
        layers.reserve(layer_generators.size());
        size3d volume_size = size_in;
        for (const auto& layer_generator : layer_generators)
        {
            layer_ptr layer = layer_generator(volume_size);
            volume_size = layer->output_size();
            layers.push_back(layer);
        }
        return std::make_shared<multi_layer_net>(layers);
    };
}

// conv 1*1, c/2 filters
// conv 1*x, c/2 filters
// conv y*1, c/2 filters
// conv 1*1, c filters
inline pre_layer bottleneck_sandwich_dims_individual(
    const size2d& filter_size,
    const pre_layer& activation_layer_intermediate,
    const pre_layer& activation_layer)
{
    assert(activation_layer);
    return [=](const size3d& size_in) -> layer_ptr
    {
        assert(size_in.depth_ % 2 == 0);
        std::size_t c = size_in.depth_;
        size2d filter_size_x(1, filter_size.width_);
        size2d filter_size_y(filter_size.height_, 1);
        size3d size_intermediate(c / 2, size_in.height_, size_in.width_);
        pre_layers layers = {
            conv(size2d(1, 1), c / 2, 1), activation_layer_intermediate,
            conv(filter_size_x, c / 2, 1), activation_layer_intermediate,
            conv(filter_size_y, c / 2, 1), activation_layer_intermediate,
            conv(size2d(1, 1), c, 1), activation_layer
        };
        return net(layers)(size_in);
    };
}

} // namespace fd
