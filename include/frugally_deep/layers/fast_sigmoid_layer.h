// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class fast_sigmoid_layer : public activation_layer
{
public:
    fast_sigmoid_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto activation_function = [](float_t x) -> float_t
        {
            return x / (1 + std::abs(x));
        };
        return transform_matrix3d(activation_function, in_vol);
    }
};

} // namespace fd
