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
        filters_(fplus::transform(generate_im2col_single_filter_matrix,
            generate_filters(dilation_rate, filter_shape,
                input_depth, depthwise_weights, bias))),
        strides_(strides),
        padding_(p)
    {
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
        assertion(filters_.size() == input_depth, "invalid filter shape");
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);

        const auto input_slices = tensor_to_depth_slices(input);

        assertion(input_slices.size() == filters_.size(),
            "invalid input depth");

        const auto convolve_slice =
            [&](const tensor& slice, const convolution_filter_matrices& f) -> tensor
        {
            assertion(f.filter_shape_.depth_ == 1, "invalid filter depth");
            const auto result = convolve(strides_, padding_, f, slice);
            assertion(result.shape().depth_ == 1, "invalid conv output");
            return result;
        };

        assertion(input_slices.size() == filters_.size(),
            "invalid depthwise filter count");
        return {concatenate_tensors_depth(fplus::zip_with(
            convolve_slice, input_slices, filters_))};
    }

    std::vector<convolution_filter_matrices> filters_;
    shape2 strides_;
    padding padding_;
};

} } // namespace fdeep, namespace internal
