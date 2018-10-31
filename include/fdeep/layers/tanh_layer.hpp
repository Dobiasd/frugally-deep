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

class tanh_layer : public activation_layer
{
public:
    explicit tanh_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor5 transform_input(const tensor5& in_vol) const override
    {
        const auto activation_function = [](float_type x) -> float_type
        {
            return std::tanh(x);
        };
        return transform_tensor5(activation_function, in_vol);
    }
};

} } // namespace fdeep, namespace internal
