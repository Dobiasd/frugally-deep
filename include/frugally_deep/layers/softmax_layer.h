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
        in_vol_max_ = fplus::maximum(in_vol.as_vector());

        const auto activation_function = [this](float_t x) -> float_t
        {
            return std::exp(x - in_vol_max_);
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

    matrix3d transform_error_backward_pass(const matrix3d&) const override
    {
        assert(false); // not implemented yet
    }
};

} // namespace fd
