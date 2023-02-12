// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class resizing_layer : public layer
{
public:
    explicit resizing_layer(const std::string& name,
        std::size_t height, std::size_t width,
        const std::string& interpolation, bool crop_to_aspect_ratio) :
    layer(name),
    height_(height),
    width_(width),
    interpolation_(interpolation),
    crop_to_aspect_ratio_(crop_to_aspect_ratio)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);
        if (crop_to_aspect_ratio_)
        {
            return {smart_resize_tensor_2d(input, shape2(height_, width_), interpolation_)};
        }
        else
        {
            return {resize_tensor_2d(input, shape2(height_, width_), interpolation_)};
        }
    }
    std::size_t height_;
    std::size_t width_;
    std::string interpolation_;
    bool crop_to_aspect_ratio_;
};

} } // namespace fdeep, namespace internal
