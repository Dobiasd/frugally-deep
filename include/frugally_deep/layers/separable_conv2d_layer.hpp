// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.hpp"
#include "frugally_deep/convolution_transpose.hpp"
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
    enum class padding { valid, same };

    explicit separable_conv2d_layer(
            const std::string& name, std::size_t input_depth,
            const shape3& filter_size,
            std::size_t k, const shape2& strides, padding p,
            const float_vec& weights_0, const float_vec& weights_1,
            const float_vec& bias_0,
            const float_vec& bias)
        : layer(name),
        filters_0_(generate_filters(
            filter_size, input_depth, weights_0, bias_0)),
        filters_1_(generate_filters(
            shape3(input_depth, 1, 1), k, weights_1, bias)),
        padding_(p),
        strides_(strides)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_size.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
        assertion(strides.width_ == strides.height_, "invalid strides");
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");
        const auto& input = inputs.front();

        assertion(strides_.width_ == strides_.height_, "invalid strides");
        const std::size_t stride = strides_.width_;

        assertion(filters_0_.size() > 0, "no filters");
        const auto filter_size = filters_0_.front().shape();

        // todo: same as in conv2d layer
        std::size_t padding_top = 0;
        std::size_t padding_bottom = 0;
        std::size_t padding_left = 0;
        std::size_t padding_right = 0;
        if (padding_ == padding::same)
        {
            padding_top = (input.shape().height_ * stride - input.shape().height_ + filter_size.height_ - stride) / 2;
            padding_left = (input.shape().width_ * stride - input.shape().width_ + filter_size.width_ - stride) / 2;
            padding_bottom = padding_top + (filter_size.height_ % 2 == 0 ? 1 : 0);
            padding_right = padding_left + (filter_size.width_ % 2 == 0 ? 1 : 0);
        }
        const auto temp = convolve(stride,
            padding_top, padding_bottom, padding_left, padding_right,
            filters_0_, input);
        return {convolve(stride, 0, 0, 0, 0, filters_1_, temp)};
    }
    filter_vec filters_0_;
    filter_vec filters_1_;
    padding padding_;
    shape2 strides_;
};

} } // namespace fdeep, namespace internal
