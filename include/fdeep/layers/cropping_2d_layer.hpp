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
        std::size_t right_crop,
        std::size_t output_dimensions) :
            layer(name),
            top_crop_(top_crop),
            bottom_crop_(bottom_crop),
            left_crop_(left_crop),
            right_crop_(right_crop),
            output_dimensions_(output_dimensions)
    {
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();

        const auto result = crop_tensor5(top_crop_, bottom_crop_, left_crop_, right_crop_, input);
        if (output_dimensions_ == 1)
        {
            // To support correct output rank for 1d version of layer.
            assertion(result.shape().rank_ == 3, "Invalid rank of conv output");
            return {tensor5_with_changed_rank(result, 2)};
        }
        return {result};
    }
    std::size_t top_crop_;
    std::size_t bottom_crop_;
    std::size_t left_crop_;
    std::size_t right_crop_;
    std::size_t output_dimensions_;
};

} } // namespace fdeep, namespace internal
