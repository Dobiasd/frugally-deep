// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape_hw.hpp"
#include "fdeep/shape_hwc.hpp"
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
            const shape_hwc& filter_shape,
            std::size_t k, const shape_hw& strides, padding p,
            bool padding_valid_offset_depth_1,
            bool padding_same_offset_depth_1,
            bool padding_valid_offset_depth_2,
            bool padding_same_offset_depth_2,
            const shape_hw& dilation_rate,
            const float_vec& depthwise_weights,
            const float_vec& pointwise_weights,
            const float_vec& bias_0,
            const float_vec& bias)
        : layer(name),
        filters_depthwise_(fplus::transform(generate_im2col_single_filter_matrix,
            generate_filters(dilation_rate, filter_shape,
                input_depth, depthwise_weights, bias_0))),
        filters_pointwise_(generate_im2col_filter_matrix(
            generate_filters(shape_hw(1, 1),
                shape_hwc(1, 1, input_depth), k, pointwise_weights, bias))),
        strides_(strides),
        padding_(p),
        padding_valid_offset_depth_1_(padding_valid_offset_depth_1),
        padding_same_offset_depth_1_(padding_same_offset_depth_1),
        padding_valid_offset_depth_2_(padding_valid_offset_depth_2),
        padding_same_offset_depth_2_(padding_same_offset_depth_2)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
        assertion(filters_depthwise_.size() == input_depth,
            "invalid number of filters");
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");

        const auto input_slices = tensor3_to_tensor_2_depth_slices(inputs.front());

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid input depth");

        const bool use_offset = input_slices.size() == 1 ?
            ((padding_ == padding::valid && padding_valid_offset_depth_1_) ||
            (padding_ == padding::same && padding_same_offset_depth_1_)) :
            ((padding_ == padding::valid && padding_valid_offset_depth_2_) ||
            (padding_ == padding::same && padding_same_offset_depth_2_));

        const auto convolve_slice =
            [&](const tensor2& slice, const im2col_filter_matrix& f) -> tensor3
        {
            assertion(f.filter_shape_.depth_ == 1, "invalid filter depth");
            const auto result = convolve(strides_, padding_,
                use_offset, f, tensor2_to_tensor3(slice));
            assertion(result.shape().depth_ == 1, "invalid conv output");
            return result;
        };

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid depthwise filter count");
        const auto temp = concatenate_tensor3s_depth(fplus::zip_with(
            convolve_slice, input_slices, filters_depthwise_));

        return {convolve(shape_hw(1, 1), padding::valid, false,
            filters_pointwise_, temp)};
    }

    std::vector<im2col_filter_matrix> filters_depthwise_;
    im2col_filter_matrix filters_pointwise_;
    shape_hw strides_;
    padding padding_;
    bool padding_valid_offset_depth_1_;
    bool padding_same_offset_depth_1_;
    bool padding_valid_offset_depth_2_;
    bool padding_same_offset_depth_2_;
};

} } // namespace fdeep, namespace internal
