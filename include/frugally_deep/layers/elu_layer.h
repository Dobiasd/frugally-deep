// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/actication_layer.h"

namespace fd
{

class elu_layer : public actication_layer
{
public:
    explicit elu_layer(float_t alpha) : alpha_(alpha)
    {
    }
private:
    float_t alpha_;
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto actication_function = [this](float_t x) -> float_t
        {
            return x > 0 ? x : alpha_ * (std::exp(x) - 1);
        };
        return transform_helper(actication_function, in_vol);
    }
};

} // namespace fd
