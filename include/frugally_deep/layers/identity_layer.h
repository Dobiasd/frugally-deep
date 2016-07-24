// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fd
{

class identity_layer : public activation_layer
{
public:
    explicit identity_layer(const size3d& size_in)
        : activation_layer(size_in)
    {
    }
protected:
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        return in_vol;
    }
    matrix3d transform_error_backward_pass(const matrix3d& e) const override
    {
        return e;
    }
};

} // namespace fd
