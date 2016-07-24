// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/pool_layer.h"

#include <limits>

namespace fd
{

class max_pool_layer : public pool_layer
{
public:
    explicit max_pool_layer(const size3d& size_in, std::size_t scale_factor) :
        pool_layer(size_in, scale_factor)
    {
    }
protected:
    matrix3d pool(const matrix3d& in_vol) const override
    {
        float_t pool_helper_acc_init = std::numeric_limits<float>::min();
        auto pool_helper_max = [](float_t& acc, float_t val) -> void
        {
            acc = std::max(acc, val);
        };
        auto pool_helper_identity = [this](float_t acc) -> float_t
        {
            return acc;
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            pool_helper_max,
            pool_helper_identity,
            in_vol);
    }
    matrix3d pool_backwards(const matrix3d& input,
        float_vec&) const override
    {
        const auto& fill_out_vol_square = [this](
            std::size_t z,
            std::size_t y,
            std::size_t x,
            float_t err_val,
            matrix3d& out_vol)
        {
            matrix3d_pos max_pos(0, 0, 0);
            float_t max_val = std::numeric_limits<float_t>::min();
            for (std::size_t yf = 0; yf < scale_factor_; ++yf)
            {
                for (std::size_t xf = 0; xf < scale_factor_; ++xf)
                {
                    matrix3d_pos last_input_pos(z, y + yf, x + xf);
                    const float_t val = last_input_.get(last_input_pos);
                    if (val > max_val)
                    {
                        max_val = val;
                        max_pos = last_input_pos;
                    }
                }
            }
            out_vol.set(max_pos, err_val);
        };
        return pool_backwards_helper(fill_out_vol_square, input);
    }
};

} // namespace fd
