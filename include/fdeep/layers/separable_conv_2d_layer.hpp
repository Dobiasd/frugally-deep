// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape2.hpp"
#include "fdeep/tensor_shape.hpp"
#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

// Convolve depth slices separately first.
// Then convolve normally with kernel_size = (1, 1)
class separable_conv_2d_layer : public layer
{
public:
    explicit separable_conv_2d_layer(
            const std::string& name, std::size_t input_depth,
            const tensor_shape& filter_shape,
            std::size_t k, const shape2& strides, padding p,
            const shape2& dilation_rate,
            const float_vec& depthwise_weights,
            const float_vec& pointwise_weights,
            const float_vec& bias_0,
            const float_vec& bias)
        : layer(name),
        depthwise_layer_(name + "_depthwise_part", input_depth,
            filter_shape, strides, p, dilation_rate,
            depthwise_weights, bias_0),
        filters_pointwise_(generate_im2col_filter_matrix(
            generate_filters(shape2(1, 1),
                tensor_shape(input_depth), k, pointwise_weights, bias)))
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto temp = depthwise_layer_.apply(inputs);
        const auto temp_single = single_tensor_from_tensors(temp);
        return {convolve(shape2(1, 1), padding::valid, filters_pointwise_, temp_single)};
    }

    depthwise_conv_2d_layer depthwise_layer_;
    convolution_filter_matrices filters_pointwise_;
};

} } // namespace fdeep, namespace internal
