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
    explicit tanh_layer(const std::string& name,
        bool snd_deriv_max_at_1,
        float_t alpha)
        : activation_layer(name),
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
        return 1.7519f * std::tanh(2 * x / 3) + alpha * x;
    };

    static float_t activation_function_def_deriv(float_t alpha, float_t x)
    {
        return alpha + 1 - fplus::square(activation_function_def(0, x));
    };

    static float_t sech(float_t x)
    {
        return 1 / static_cast<float_t>(cosh(x));
    };
    static float_t activation_function_snd_deriv_max_at_1_deriv(float_t alpha, float_t x)
    {
        return alpha + (1.7519f * 2 / 3) *
            static_cast<float_t>(fplus::square(sech(2 * x / 3)));
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
};

} // namespace fd
