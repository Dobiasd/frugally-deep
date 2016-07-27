// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

// snd_deriv_max_at_1 == true
//     switches function from
//     f(x) = tanh(x);
//     to
//     f(x) = 1.7519 * tanh(2 * x / 3);
//     according to "Efficient BackProp" (LeCun et al.)
class tanh_layer : public activation_layer
{
public:
    explicit tanh_layer(const size3d& size_in,
        bool snd_deriv_max_at_1,
        float_t alpha)
        : activation_layer(size_in),
        snd_deriv_max_at_1_(snd_deriv_max_at_1),
        alpha_(alpha)
    {
    }
protected:
    const bool snd_deriv_max_at_1_;
    const float_t alpha_;

    static float_t activation_function_def(float_t alpha, float_t x)
    {
        return std::tanh(x) + alpha * x;
    };

    static float_t activation_function_snd_deriv_max_at_1(float_t alpha, float_t x)
    {
        return 1.7519 * std::tanh(2 * x / 3) + alpha * x;
    };

    static float_t activation_function_def_deriv(float_t alpha, float_t x)
    {
        return alpha + 1 - fplus::square(activation_function_def(0, x));
    };

    static float_t sech(float_t x)
    {
        return 1 / cosh(x);
    };
    static float_t activation_function_snd_deriv_max_at_1_deriv(float_t alpha, float_t x)
    {
        return alpha + (1.7519 * 2 / 3) * fplus::square(sech(2 * x / 3));
    };

    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        return snd_deriv_max_at_1_
            ? transform_matrix3d(
                [this](float_t x) -> float_t
                {
                    return activation_function_snd_deriv_max_at_1(alpha_, x);
                }, in_vol)
            : transform_matrix3d(
                [this](float_t x) -> float_t
                {
                    return activation_function_def(alpha_, x);
                }, in_vol);
    }
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        const auto last_input_derivs = snd_deriv_max_at_1_
            ? transform_matrix3d(
                [this](float_t x) -> float_t
                {
                    return activation_function_snd_deriv_max_at_1_deriv(alpha_, x);
                }, last_input_)
            : transform_matrix3d(
                [this](float_t x) -> float_t
                {
                    return activation_function_def_deriv(alpha_, x);
                }, last_input_);
        return multiply_matrix3ds_elementwise(last_input_derivs, e);
    }
};

} // namespace fd
