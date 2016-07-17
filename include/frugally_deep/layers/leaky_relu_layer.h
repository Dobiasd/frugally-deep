// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/actication_layer.h"

namespace fd
{

class leaky_relu_layer : public actication_layer
{
public:
    explicit leaky_relu_layer(float_t alpha) : alpha_(alpha)
    {
    }
private:
    float_t alpha_;
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto actication_function = [this](float_t x) -> float_t
        {
            return x > 0 ? x : alpha_ * x;
        };
        return transform_matrix3d(actication_function, in_vol);
    }
};

} // namespace fd
