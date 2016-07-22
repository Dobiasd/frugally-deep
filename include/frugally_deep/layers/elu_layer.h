// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class elu_layer : public activation_layer
{
public:
    explicit elu_layer(const size3d& size_in, float_t alpha)
        : activation_layer(size_in), alpha_(alpha)
    {
    }
protected:
    const float_t alpha_;
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto activation_function = [this](float_t x) -> float_t
        {
            return x > 0 ? x : alpha_ * (std::exp(x) - 1);
        };
        return transform_matrix3d(activation_function, in_vol);
    }
};

} // namespace fd
