// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class softmax_layer : public activation_layer
{
public:
    explicit softmax_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    matrix3d transform_input(const matrix3d& input) const override
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

        const auto unnormalized_sum = fplus::sum(unnormalized.as_vector());
        const auto div_by_unnormalized_sum =
            [unnormalized_sum](float_t x) -> float_t
        {
            return x / unnormalized_sum;
        };

        return {transform_matrix3d(div_by_unnormalized_sum, unnormalized)};
    }
};

} // namespace fd
