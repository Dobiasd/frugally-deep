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
class conv2d_layer : public layer
{
public:
    enum class padding { valid, same };

    explicit conv2d_layer(
            const std::string& name, const shape3& filter_size,
            std::size_t k, const shape2& strides, padding p,
            const float_vec& weights, const float_vec& bias)
        : layer(name),
        filters_(generate_filters(filter_size, k, weights, bias)),
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

        assertion(filters_.size() > 0, "no filters");
        const auto filter_size = filters_.front().size();

        std::size_t padding_y_ = 0;
        std::size_t padding_x_ = 0;
        if (padding_ == padding::same)
        {
            // todo: is this correct?
            padding_y_ = (input.size().height_ * stride - input.size().height_ + filter_size.height_ - stride) / 2;
            padding_x_ = (input.size().width_ * stride - input.size().width_ + filter_size.width_ - stride) / 2;
        }

        return {convolve(stride, padding_x_, padding_y_, filters_, input)};
    }
    filter_vec filters_;
    padding padding_;
    shape2 strides_;
};

} } // namespace fdeep, namespace internal
