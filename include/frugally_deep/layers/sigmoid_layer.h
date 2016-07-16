// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/actication_layer.h"

namespace fd
{

// fast sigmoid: f(x) = x / (1 + abs(x))
class sigmoid_layer : public actication_layer
{
private:
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto actication_function = [](float_t x) -> float_t
        {
            return x / (1 + std::abs(x));
        };
        return transform_helper(actication_function, in_vol);
    }
};

} // namespace fd
