// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/actication_layer.h"

namespace fd
{

class softmax_layer : public actication_layer
{
public:
    explicit softmax_layer(const size3d& size_in)
        : actication_layer(size_in)
    {
    }
private:
    matrix3d transform_input(const matrix3d& in_vol) const override
    {
        // http://stackoverflow.com/q/9906136/1866775
        const auto& in_vol_values = in_vol.as_vector();
        float_t in_vol_max = fplus::maximum(in_vol_values);
        auto actication_function = [in_vol_max](float_t x) -> float_t
        {
            return std::exp(x - in_vol_max);
        };
        auto unnormalized = transform_matrix3d(actication_function, in_vol);

        auto unnormalized_sum = fplus::sum(unnormalized.as_vector());
        auto make_sum_equal_to_one = [unnormalized_sum](float_t x) -> float_t
        {
            return x / unnormalized_sum;
        };
        return transform_matrix3d(make_sum_equal_to_one, unnormalized);
    }
};

} // namespace fd
