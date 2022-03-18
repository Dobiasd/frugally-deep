// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/depthwise_convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape2.hpp"
#include "fdeep/tensor_shape.hpp"
#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

// Convolve depth slices separately.
class depthwise_conv_2d_layer : public layer
{
public:
    explicit depthwise_conv_2d_layer(
            const std::string& name, std::size_t input_depth,
            const tensor_shape& filter_shape,
            const shape2& strides, padding p,
            const shape2& dilation_rate,
            const float_vec& depthwise_weights,
            const float_vec& bias)
        : layer(name),
        filters_(generate_im2col_filter_matrix(
            generate_filters(dilation_rate, filter_shape,
                input_depth, depthwise_weights, bias))),
        strides_(strides),
        padding_(p)
    {
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
        assertion(filters_.filter_count_ == input_depth,
            "invalid number of filters");
        assertion(filters_.filter_shape_.depth_ == 1,
            "invalid filter shape");
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        const auto result = depthwise_convolve(strides_, padding_, filters_, input);
        assertion(result.shape().depth_ == input.shape().depth_,
            "Invalid output shape");
        return {result};
    }

    convolution_filter_matrices filters_;
    shape2 strides_;
    padding padding_;
};

} } // namespace fdeep, namespace internal
