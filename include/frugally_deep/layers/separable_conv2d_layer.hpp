// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.hpp"
#include "frugally_deep/filter.hpp"
#include "frugally_deep/shape2.hpp"
#include "frugally_deep/shape3.hpp"
#include "frugally_deep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

// todo: variable padding, variable strides
// Convolve depth slices separately first.
// Then convolve normally with kernel_size = (1, 1)
class separable_conv2d_layer : public layer
{
public:
    explicit separable_conv2d_layer(
            const std::string& name, std::size_t input_depth,
            const shape3& filter_shape,
            std::size_t k, const shape2& strides, padding p,
            bool padding_valid_uses_offset, bool padding_same_uses_offset,
            const float_vec& depthwise_weights,
            const float_vec& pointwise_weights,
            const float_vec& bias_0,
            const float_vec& bias)
        : layer(name),
        filters_depthwise_(generate_filters(filter_shape,
            input_depth, depthwise_weights, bias_0)),
        filters_pointwise_(generate_filters(shape3(input_depth, 1, 1),
            k, pointwise_weights, bias)),
        strides_(strides),
        padding_(p),
        padding_valid_uses_offset_(padding_valid_uses_offset),
        padding_same_uses_offset_(padding_same_uses_offset)
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

        const auto input_slices = tensor3_to_depth_slices(inputs.front());

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid input depth");

        const bool use_offset =
            input_slices.size() == 1 && // todo: verify this in export script
            ((padding_ == padding::valid && padding_valid_uses_offset_) ||
            (padding_ == padding::same && padding_same_uses_offset_));

        const auto convolve_slice =
            [&](const tensor2& slice, const filter& f) -> tensor3
        {
            assertion(f.shape().depth_ == 1, "invalid filter depth");
            const auto result = convolve(strides_, padding_,
                use_offset, filter_vec(1, f), tensor2_to_tensor3(slice));
            assertion(result.shape().depth_ == 1, "invalid conv output");
            return result;
        };

        assertion(input_slices.size() == filters_depthwise_.size(),
            "invalid depthwise filter count");
        const auto temp = concatenate_tensor3s(fplus::zip_with(
            convolve_slice, input_slices, filters_depthwise_));

        return {convolve(shape2(1, 1), padding::valid, false,
            filters_pointwise_, temp)};
    }
    filter_vec filters_depthwise_;
    filter_vec filters_pointwise_;
    shape2 strides_;
    padding padding_;
    bool padding_valid_uses_offset_;
    bool padding_same_uses_offset_;
};

} } // namespace fdeep, namespace internal
