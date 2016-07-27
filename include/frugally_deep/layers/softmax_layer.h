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

        const auto ex = [this](float_t x) -> float_t
        {
            return std::exp(x);
            //return std::exp(x - in_vol_max_);
        };

        const auto unnormalized = transform_matrix3d(ex, in_vol);

        unnormalized_sum_ = fplus::sum(unnormalized.as_vector());
        const auto div_by_unnormalized_sum = [this](float_t x) -> float_t
        {
            return x / unnormalized_sum_;
        };

        last_output_ = transform_matrix3d(div_by_unnormalized_sum, unnormalized);
        return last_output_;
    }

    matrix3d transform_error_backward_pass(const matrix3d& fb) const override
    {
        const auto fb_vec = fb.as_vector();
        const float_vec li_vec = last_input_.as_vector();
        float_vec fa_vec(input_size().volume(), 0);


        const auto ex = [this](float_t x) -> float_t
        {
            return std::exp(x);
            //return std::exp(x - in_vol_max_);
        };

        for (std::size_t i = 0; i < fa_vec.size(); ++i)
        {
            for (std::size_t j = 0; j < fb_vec.size(); ++j)
            {
                if (j == i)
                {
                    const float_t x = li_vec[i];
                    const float_t ex_x = ex(x);
                    const float_t del_sigma_i =
                        ex_x * (unnormalized_sum_ - ex_x) /
                            fplus::square(unnormalized_sum_);
                    fa_vec[i] += del_sigma_i * fb_vec[j];
                }
                else
                {
                    const float_t x = li_vec[i];
                    const float_t y = li_vec[j];
                    const float_t ex_x_plus_y = ex(x + y);
                    const float_t del_sigma_j =
                        -ex_x_plus_y /
                            fplus::square(unnormalized_sum_);
                    fa_vec[i] += del_sigma_j * fb_vec[j];
                }
            }
        }

        return matrix3d(input_size(), fa_vec);
    }
};

} // namespace fd
