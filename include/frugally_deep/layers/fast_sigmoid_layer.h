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
    fast_sigmoid_layer(const size3d& size_in)
        : activation_layer(size_in)
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
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        auto activation_function_deriv = [this](float_t x) -> float_t
        {
            return 1 / fplus::square(std::abs(x) + 1);
        };
        const auto last_input_derivs =
            transform_matrix3d(activation_function_deriv, last_input_);
        return multiply_matrix3ds_elementwise(last_input_derivs, e);
    }
};

} // namespace fd
