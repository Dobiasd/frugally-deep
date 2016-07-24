// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class softmax_layer : public activation_layer
{
public:
    explicit softmax_layer(const size3d& size_in)
        : activation_layer(size_in),
        in_vol_max_(0),
        unnormalized_sum_(0)
    {
    }
protected:
    mutable float_t in_vol_max_;
    mutable float_t unnormalized_sum_;
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        // http://stackoverflow.com/q/9906136/1866775
        const auto& in_vol_values = in_vol.as_vector();
        in_vol_max_ = fplus::maximum(in_vol_values);
        auto activation_function = [this](float_t x) -> float_t
        {
            return std::exp(x - in_vol_max_);
        };
        auto unnormalized = transform_matrix3d(activation_function, in_vol);

        unnormalized_sum_ = fplus::sum(unnormalized.as_vector());
        auto make_sum_equal_to_one = [this](float_t x) -> float_t
        {
            return x / unnormalized_sum_;
        };
        return transform_matrix3d(make_sum_equal_to_one, unnormalized);
    }
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        assert(false);
        auto activation_function_deriv = [this](float_t x) -> float_t
        {
            const auto y = std::exp(x - in_vol_max_) / unnormalized_sum_;
            return y * (1 - y);
        };
        const auto last_input_derivs =
            transform_matrix3d(activation_function_deriv, last_input_);
        return multiply_matrix3ds_elementwise(last_input_derivs, e);
    }
};

} // namespace fd
