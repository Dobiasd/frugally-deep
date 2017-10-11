// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape2.hpp"
#include "fdeep/shape3.hpp"
#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

class conv2d_layer : public layer
{
public:
    explicit conv2d_layer(
            const std::string& name, const shape3& filter_shape,
            std::size_t k, const shape2& strides, padding p,
            bool padding_valid_uses_offset, bool padding_same_uses_offset,
            const float_vec& weights, const float_vec& bias)
        : layer(name),
        filters_(generate_filters(filter_shape, k, weights, bias)),
        strides_(strides),
        padding_(p),
        padding_valid_uses_offset_(padding_valid_uses_offset),
        padding_same_uses_offset_(padding_same_uses_offset)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
    }
protected:
    tensor3s apply_impl(bool use_im2col, const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");
        bool use_offset =
            (padding_ == padding::valid && padding_valid_uses_offset_) ||
            (padding_ == padding::same && padding_same_uses_offset_);
        return {convolve(strides_, padding_, use_offset,
            filters_, inputs.front(), use_im2col)};
    }
    filter_vec filters_;
    shape2 strides_;
    padding padding_;
    bool padding_valid_uses_offset_;
    bool padding_same_uses_offset_;
};

} } // namespace fdeep, namespace internal
