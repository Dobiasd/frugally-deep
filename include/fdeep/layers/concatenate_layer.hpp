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

class concatenate_layer : public layer
{
public:
    explicit concatenate_layer(const std::string& name, std::int32_t axis)
        : layer(name), axis_(axis)
    {
    }
protected:
    std::int32_t keras_axis_to_fdeep_axis(std::int32_t keras_axis)
    {
        if (keras_axis == 1)
        {
            return 1;
        }
        else if (keras_axis == 2)
        {
            return 2;
        }
        assertion(keras_axis == -1 || keras_axis == 3, "Invalid Keras axis (" + std::to_string(keras_axis) +
            ") for concatenate layer.");
        return 0;
    }
    tensors apply_impl(const tensors& input) const override
    {
        return {concatenate_tensors(input, axis_)};
    }
    std::int32_t axis_;
};

} } // namespace fdeep, namespace internal
