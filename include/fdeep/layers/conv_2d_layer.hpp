// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape2.hpp"
#include "fdeep/shape5.hpp"
#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class conv_2d_layer : public layer
{
public:
    explicit conv_2d_layer(
            const std::string& name, const shape5& filter_shape,
            std::size_t k, const shape2& strides, padding p,
            bool padding_valid_offset_depth_1,
            bool padding_same_offset_depth_1,
            bool padding_valid_offset_depth_2,
            bool padding_same_offset_depth_2,
            const shape2& dilation_rate,
            const float_vec& weights, const float_vec& bias)
        : layer(name),
        filters_(generate_im2col_filter_matrix(
            generate_filters(dilation_rate, filter_shape, k, weights, bias))),
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
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");
        const bool use_offset = inputs.front().shape().depth_ == 1 ?
            ((padding_ == padding::valid && padding_valid_offset_depth_1_) ||
            (padding_ == padding::same && padding_same_offset_depth_1_)) :
            ((padding_ == padding::valid && padding_valid_offset_depth_2_) ||
            (padding_ == padding::same && padding_same_offset_depth_2_));
        return {convolve(strides_, padding_, use_offset,
            filters_, inputs.front())};
    }
    im2col_filter_matrix filters_;
    shape2 strides_;
    padding padding_;
    bool padding_valid_offset_depth_1_;
    bool padding_same_offset_depth_1_;
    bool padding_valid_offset_depth_2_;
    bool padding_same_offset_depth_2_;
};

} } // namespace fdeep, namespace internal
