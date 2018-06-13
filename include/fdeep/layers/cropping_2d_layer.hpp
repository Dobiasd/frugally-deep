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

class cropping_2d_layer : public layer
{
public:
    explicit cropping_2d_layer(const std::string& name,
        std::size_t top_crop,
        std::size_t bottom_crop,
        std::size_t left_crop,
        std::size_t right_crop) :
            layer(name),
            top_crop_(top_crop),
            bottom_crop_(bottom_crop),
            left_crop_(left_crop),
            right_crop_(right_crop)
    {
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {crop_tensor3(top_crop_, bottom_crop_, left_crop_, right_crop_,
            input)};
    }
    std::size_t top_crop_;
    std::size_t bottom_crop_;
    std::size_t left_crop_;
    std::size_t right_crop_;
};

} } // namespace fdeep, namespace internal
