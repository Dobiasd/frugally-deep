// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/pool_layer.h"

namespace fd
{

class avg_pool_layer : public pool_layer
{
public:
    explicit avg_pool_layer(const size3d& size_in, std::size_t scale_factor) :
            pool_layer(size_in, scale_factor)
    {
    }
protected:
    matrix3d pool(const matrix3d& in_vol) const override
    {
        float_t pool_helper_acc_init = 0;
        auto pool_helper_add = [](float_t& acc, float_t val) -> void
        {
            acc += val;
        };
        auto pool_helper_div = [this](float_t acc) -> float_t
        {
            return acc / static_cast<float_t>(scale_factor_ * scale_factor_);
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            pool_helper_add,
            pool_helper_div,
            in_vol);
    }
};

} // namespace fd
