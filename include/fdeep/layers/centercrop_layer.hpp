// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class centercrop_layer : public layer
{
public:
    explicit centercrop_layer(const std::string& name,
        std::size_t height,
        std::size_t width) :
            layer(name),
            height_(height),
            width_(width)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        auto input = single_tensor_from_tensors(inputs);
        if (input.shape().height_ < height_ || input.shape().width_ < width_)
        {
            input = smart_resize_tensor_2d(input, shape2(height_, width_), "bilinear");
        }
        const std::size_t excess_height = input.shape().height_ - height_;
        const std::size_t excess_width = input.shape().width_ - width_;
        const std::size_t top_crop = excess_height / 2;
        const std::size_t left_crop = excess_width / 2;
        const std::size_t bottom_crop = excess_height - top_crop;
        const std::size_t right_crop = excess_width - left_crop;
        return {crop_tensor(0, 0, top_crop, bottom_crop, left_crop, right_crop, input)};
    }
    std::size_t height_;
    std::size_t width_;
};

} } // namespace fdeep, namespace internal
