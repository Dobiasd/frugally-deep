// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class softmax_layer : public layer
{
public:
    explicit softmax_layer(const std::string& name)
        : layer(name),
        in_vol_max_(0),
        unnormalized_sum_(0)
    {
    }

    matrix3d apply(const matrix3d& input) const override
    {
        const auto ex = [this](float_t x) -> float_t
        {
            // todo: why does this trick make the gradient check fail?
            // http://stackoverflow.com/q/9906136/1866775
            //in_vol_max_ = fplus::maximum(input.as_vector());
            //return std::exp(x - in_vol_max_);
            return std::exp(x);
        };

        const auto unnormalized = transform_matrix3d(ex, input);

        unnormalized_sum_ = fplus::sum(unnormalized.as_vector());
        const auto div_by_unnormalized_sum = [this](float_t x) -> float_t
        {
            return x / unnormalized_sum_;
        };

        return transform_matrix3d(div_by_unnormalized_sum, unnormalized);
    }

protected:
    mutable float_t in_vol_max_;
    mutable float_t unnormalized_sum_;
};

} // namespace fd
