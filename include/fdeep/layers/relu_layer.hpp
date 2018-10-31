// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <algorithm>
#include <string>

namespace fdeep { namespace internal
{

class relu_layer : public activation_layer
{
public:
    explicit relu_layer(const std::string& name, const float_type max_value)
        : activation_layer(name), max_value_(max_value)
    {
    }
protected:
    tensor5 transform_input(const tensor5& in_vol) const override
    {
        auto activation_function = [&](float_type x) -> float_type
        {
            return std::min<float_type>(std::max<float_type>(x, 0), max_value_);
        };
        return transform_tensor5(activation_function, in_vol);
    }
    float_type max_value_;
};

} } // namespace fdeep, namespace internal
