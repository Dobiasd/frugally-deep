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
        auto pool_helper_add = [](float_t acc, float_t val) -> float_t
        {
            return acc + val;
        };
        auto pool_helper_dummy = [](float_t, float_t) -> float_t { return 0; };
        auto pool_helper_div = [this](float_t acc, float_t) -> float_t
        {
            return acc / fplus::square(scale_factor_);
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            0,
            pool_helper_add,
            pool_helper_dummy,
            pool_helper_div,
            in_vol);
    }
    matrix3d pool_backwards(const matrix3d& input,
        float_vec&) const override
    {
        const float_t area = fplus::square(scale_factor_);
        const auto fill_out_vol_square = [this, area](
            std::size_t z,
            std::size_t y,
            std::size_t x,
            float_t err_val,
            matrix3d& out_vol)
        {
            for (std::size_t yf = 0; yf < scale_factor_; ++yf)
            {
                for (std::size_t xf = 0; xf < scale_factor_; ++xf)
                {
                    out_vol.set(z, y + yf, x + xf, err_val / area);
                }
            }
        };
        return pool_backwards_helper(fill_out_vol_square, input);
    }
};

} // namespace fd
