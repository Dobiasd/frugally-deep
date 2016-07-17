// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class relu_layer : public activation_layer
{
public:
    explicit relu_layer(const size3d& size_in)
        : activation_layer(size_in)
    {
    }
protected:
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        auto actication_function = [](float_t x) -> float_t
        {
            return std::max<float_t>(x, 0);
        };
        return transform_matrix3d(actication_function, in_vol);
    }
};

} // namespace fd
