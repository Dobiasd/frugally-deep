// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.h"
#include "frugally_deep/convolution_transpose.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// upconvolution layer
// aka. convolution transpose
// aka. backward strided convolution
// aka. fractionally strided convolution
// aka. deconvolution
class convolution_transpose_layer : public layer
{
public:
    enum class padding { valid, same };
    explicit convolution_transpose_layer(
            const std::string& name, const size3d& filter_size,
            std::size_t k, const size2d& strides, padding p,
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
    matrix3ds apply_impl(const matrix3ds& inputs) const override
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

        return {convolve_transpose(
            stride, padding_x_, padding_y_, filters_, input)};
    }
    filter_vec filters_;
    padding padding_;
    size2d strides_;
};

} // namespace fd
