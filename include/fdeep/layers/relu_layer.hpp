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
    explicit relu_layer(const std::string& name,
        const float_type max_value,
        const float_type negative_slope,
        const float_type threshold)
        : activation_layer(name),
        max_value_(max_value),
        negative_slope_(negative_slope),
        threshold_(threshold)
    {
    }
protected:
    tensor transform_input(const tensor& in_vol) const override
    {
        auto activation_function = [&](float_type x) -> float_type
        {
            if (x >= max_value_)
                return max_value_; 
            if (threshold_ <= x && x < max_value_)
                return x;
            return negative_slope_ * (x - threshold_);
        };
        return transform_tensor(activation_function, in_vol);
    }
    float_type max_value_;
    float_type negative_slope_;
    float_type threshold_;
};

} } // namespace fdeep, namespace internal
