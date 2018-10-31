// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class leaky_relu_layer : public activation_layer
{
public:
    explicit leaky_relu_layer(const std::string& name, float_type alpha) :
        activation_layer(name), alpha_(alpha)
    {
    }
protected:
    float_type alpha_;
    tensor5 transform_input(const tensor5& in_vol) const override
    {
        auto activation_function = [this](float_type x) -> float_type
        {
            return x > 0 ? x : alpha_ * x;
        };
        return transform_tensor5(activation_function, in_vol);
    }
};

} } // namespace fdeep, namespace internal
