// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/activation_layer.h"

namespace fdeep
{

class elu_layer : public activation_layer
{
public:
    explicit elu_layer(const std::string& name, float_t alpha)
        : activation_layer(name), alpha_(alpha)
    {
    }
protected:
    const float_t alpha_;
    static float_t activation_function(float_t alpha, float_t x)
    {
        return x >= 0 ? x : alpha * (std::exp(x) - 1);
    }
    tensor3 transform_input(const tensor3& in_vol) const override
    {
        return transform_tensor3(
            fplus::bind_1st_of_2(activation_function, alpha_),
            in_vol);
    }
};

} // namespace fdeep
