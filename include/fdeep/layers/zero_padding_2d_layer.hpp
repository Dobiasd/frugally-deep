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

class zero_padding_2d_layer : public layer
{
public:
    explicit zero_padding_2d_layer(const std::string& name,
        std::size_t top_pad,
        std::size_t bottom_pad,
        std::size_t left_pad,
        std::size_t right_pad) :
            layer(name),
            top_pad_(top_pad),
            bottom_pad_(bottom_pad),
            left_pad_(left_pad),
            right_pad_(right_pad)
    {
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {pad_tensor5(0, top_pad_, bottom_pad_, left_pad_, right_pad_,
            input)};
    }
    std::size_t top_pad_;
    std::size_t bottom_pad_;
    std::size_t left_pad_;
    std::size_t right_pad_;
};

} } // namespace fdeep, namespace internal
