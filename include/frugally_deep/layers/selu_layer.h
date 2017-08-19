// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

// https://arxiv.org/pdf/1706.02515.pdf
class selu_layer : public activation_layer
{
public:
    explicit selu_layer(const size3d& size_in)
        : activation_layer(size_in)
    {
    }
protected:
    const float_t alpha_ = 1.6732632423543772848170429916717;
    const float_t scale_ = 1.0507009873554804934193349852946;
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto activation_function = [this](float_t x) -> float_t
        {
            return scale_ * (x >= 0.0 ? x : alpha_ * std::exp(x) - alpha_);
        };
        return transform_matrix3d(activation_function, in_vol);
    }
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        auto activation_function_deriv = [this](float_t x) -> float_t
        {
            return x > 0 ? scale_ : scale_ * alpha_ * std::exp(x);
        };
        const auto last_input_derivs =
            transform_matrix3d(activation_function_deriv, last_input_);
        return multiply_matrix3ds_elementwise(last_input_derivs, e);
    }
};

} // namespace fd
