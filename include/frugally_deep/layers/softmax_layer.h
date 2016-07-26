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
        unnormalized_sum_(0),
        last_output_(size_in)
    {
    }
protected:
    mutable float_t in_vol_max_;
    mutable float_t unnormalized_sum_;
    mutable matrix3d last_output_;

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
        const auto div_by_unnormalized_sum = [this](float_t x) -> float_t
        {
            return x / unnormalized_sum_;
        };

        last_output_ = transform_matrix3d(div_by_unnormalized_sum, unnormalized);
        return last_output_;
    }


    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        matrix3d out_vol(size3d(
            e.size().depth_,
            e.size().height_,
            e.size().width_));
        for (std::size_t z_pos1 = 0; z_pos1 < e.size().depth_; ++z_pos1)
        {
            for (std::size_t y_pos1 = 0; y_pos1 < e.size().height_; ++y_pos1)
            {
                for (std::size_t x_pos1 = 0; x_pos1 < e.size().width_; ++x_pos1)
                {
                    matrix3d_pos pos1(z_pos1, y_pos1, x_pos1);
                    for (std::size_t z_pos2 = 0; z_pos2 < e.size().depth_; ++z_pos2)
                    {
                        for (std::size_t y_pos2 = 0; y_pos2 < e.size().height_; ++y_pos2)
                        {
                            for (std::size_t x_pos2 = 0; x_pos2 < e.size().width_; ++x_pos2)
                            {
                                matrix3d_pos pos2(z_pos2, y_pos2, x_pos2);
                                const float_t e_i = e.get(pos1);
                                //const float_t e_j = e.get(pos2);
                                const float_t y_i = last_output_.get(pos1);
                                const float_t y_j = last_output_.get(pos2);
                                if (pos1 == pos2)
                                {
                                    const float_t delta = e_i * y_i * (1 - y_i);
                                    out_vol.set(pos1, out_vol.get(pos1) + delta);
                                }
                                else
                                {
                                    const float_t delta = e_i * (-y_i * y_j);
                                    out_vol.set(pos1, out_vol.get(pos1) + delta);
                                }
                            }
                        }
                    }
                }
            }
        }
        return out_vol;


        const auto activation_function = [this](float_t x) -> float_t
        {
            //return std::exp(x - in_vol_max_);
            return std::exp(x);
        };
/*
        matrix3d out_vol(size3d(
            e.size().depth_,
            e.size().height_,
            e.size().width_));

        for (std::size_t z_pos = 0; z_pos < e.size().depth_; ++z_pos)
        {
            for (std::size_t y_pos = 0; y_pos < e.size().height_; ++y_pos)
            {
                for (std::size_t x_pos = 0; x_pos < e.size().width_; ++x_pos)
                {
                    const matrix3d_pos pos(z_pos, y_pos, x_pos);
                    const float_t x = last_input_.get(pos);
                    const float_t x_exp = activation_function(x);
                    const auto reduced_last_input_exp_sum =
                        unnormalized_sum_ - x_exp;
                    const float_t numerator = x_exp * reduced_last_input_exp_sum;
                    const float_t denominator = fplus::square(unnormalized_sum_);
                    out_vol.set(pos, e.get(pos) * numerator / denominator);
                }
            }
        }
        return out_vol;
*/


        //const auto errors_exp_sum =

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
