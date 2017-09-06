// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/pool_layer.h"

#include <fplus/fplus.hpp>

#include <limits>

namespace fd
{

// A (linear) blend between max pooling and average pooling.
// The value alpha [0,1] determines the weights of both parts.
// alpha = 1 -> only max pooling
// alpha = 0 -> only avg pooling
class gentle_max_pool_layer : public pool_layer
{
public:
    explicit gentle_max_pool_layer(const std::string& name,
            std::size_t scale_factor,
            float_t alpha) :
        pool_layer(name, scale_factor),
        max_weight_(alpha),
        avg_weight_(1 - alpha)
    {
        assert(fplus::is_in_closed_interval<float_t>(0, 1, alpha));
    }
protected:
    const float_t max_weight_;
    const float_t avg_weight_;
    matrix3d pool(const matrix3d& in_vol) const override
    {
        float_t pool_helper_acc_init = std::numeric_limits<float>::lowest();
        float_t pool_helper_acc2_init = 0;
        auto pool_helper_max = [&](float_t acc, float_t val) -> float_t
        {
            return std::max(acc, val);
        };
        auto pool_helper_add = [&](float_t acc, float_t val) -> float_t
        {
            return acc + val;
        };
        auto pool_helper_finalize = [this](
            float_t acc, float_t acc2) -> float_t
        {
            return max_weight_ * acc +
                avg_weight_ * acc2 / fplus::square(scale_factor_);
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            pool_helper_acc2_init,
            pool_helper_max,
            pool_helper_add,
            pool_helper_finalize,
            in_vol);
    }
};

} // namespace fd
