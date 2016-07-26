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
        //in_vol_max_ = fplus::maximum(in_vol.as_vector());

        const auto activation_function = [this](float_t x) -> float_t
        {
            //return std::exp(x - in_vol_max_);
            return std::exp(x);
        };

        const auto unnormalized = transform_matrix3d(activation_function, in_vol);

        unnormalized_sum_ = fplus::sum(unnormalized.as_vector());
        const auto make_sum_equal_to_one = [this](float_t x) -> float_t
        {
            return x / unnormalized_sum_;
        };
        return transform_matrix3d(make_sum_equal_to_one, unnormalized);
    }
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        const auto activation_function = [this](float_t x) -> float_t
        {
            //return std::exp(x - in_vol_max_);
            return std::exp(x);
        };

        //const auto errors_exp_sum =
            //fplus::sum(fplus::transform(activation_function, e.as_vector()));
/*
        matrix3d out_vol(size3d(
            in_vol.size().depth_,
            in_vol.size().height_,
            in_vol.size().width_));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < in_vol.size().height_; ++y)
            {
                for (std::size_t x = 0; x < in_vol.size().width_; ++x)
                {
                    out_vol.set(z, y, x, f(in_vol.get(z, y, x)));
                }
            }
        }
*/
        const auto activation_function_deriv = [this, activation_function](
            float_t x) -> float_t
        {
            //const float_t exp_x = std::exp(x - in_vol_max_);
            const float_t exp_x = activation_function(x);
            const auto unnormalized_sum_wo_exp_x = unnormalized_sum_ - exp_x;
            return exp_x * unnormalized_sum_wo_exp_x /
                fplus::square(unnormalized_sum_wo_exp_x);
        };
        const auto last_input_derivs =
            transform_matrix3d(activation_function_deriv, last_input_);
        return multiply_matrix3ds_elementwise(last_input_derivs, e);
    }
};

} // namespace fd
