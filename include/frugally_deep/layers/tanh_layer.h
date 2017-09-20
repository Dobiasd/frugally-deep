// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class tanh_layer : public activation_layer
{
public:
    explicit tanh_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    static float_t activation_function_def(float_t x)
    {
        return std::tanh(x);
    };
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        return transform_matrix3d(activation_function_def, in_vol);
    }
};

} // namespace fd
