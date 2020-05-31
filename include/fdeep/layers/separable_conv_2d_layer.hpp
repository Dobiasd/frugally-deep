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
        filters_depthwise_(fplus::transform(generate_im2col_single_filter_matrix,
            generate_filters(dilation_rate, filter_shape,
                input_depth, depthwise_weights, bias_0))),
        filters_pointwise_(generate_im2col_filter_matrix(
            generate_filters(shape2(1, 1),
                tensor_shape(input_depth), k, pointwise_weights, bias))),
        strides_(strides),
        padding_(p)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
        assertion(filters_depthwise_.size() == input_depth,
            "invalid number of filters");
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);

        const auto input_slices = tensor_to_depth_slices(input);

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid input depth");

        const auto convolve_slice =
            [&](const tensor& slice, const im2col_filter_matrix& f) -> tensor
        {
            assertion(f.filter_shape_.depth_ == 1, "invalid filter depth");
            const auto result = convolve(strides_, padding_, f, slice);
            assertion(result.shape().depth_ == 1, "invalid conv output");
            return result;
        };

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid depthwise filter count");
        const auto temp = concatenate_tensors_depth(fplus::zip_with(
            convolve_slice, input_slices, filters_depthwise_));

        return {convolve(shape2(1, 1), padding::valid, filters_pointwise_, temp)};
    }

    std::vector<im2col_filter_matrix> filters_depthwise_;
    im2col_filter_matrix filters_pointwise_;
    shape2 strides_;
    padding padding_;
};

} } // namespace fdeep, namespace internal
