// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/pooling_layer.h"

#include <limits>

namespace fdeep { namespace internal
{

class max_pooling_2d_layer : public pooling_layer
{
public:
    explicit max_pooling_2d_layer(const std::string& name, std::size_t scale_factor) :
        pooling_layer(name, scale_factor)
    {
    }
protected:
    tensor3 pool(const tensor3& in_vol) const override
    {
        float_t pool_helper_acc_init = std::numeric_limits<float>::lowest();
        auto pool_helper_max = [](float_t acc, float_t val) -> float_t
        {
            return std::max(acc, val);
        };
        auto pool_helper_dummy = [](float_t, float_t) -> float_t { return 0; };
        auto pool_helper_identity = [this](float_t acc, float_t) -> float_t
        {
            return acc;
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            0,
            pool_helper_max,
            pool_helper_dummy,
            pool_helper_identity,
            in_vol);
    }
};

} } // namespace fdeep, namespace internal
