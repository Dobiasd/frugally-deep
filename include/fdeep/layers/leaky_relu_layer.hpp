// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <string>

namespace fdeep {
namespace internal {

    class leaky_relu_layer : public activation_layer {
    public:
        explicit leaky_relu_layer(const std::string& name, float_type negative_slope)
            : activation_layer(name)
            , negative_slope_(negative_slope)
        {
        }

    protected:
        float_type negative_slope_;
        tensor transform_input(const tensor& in_vol) const override
        {
            auto activation_function = [this](float_type x) -> float_type {
                return x > 0 ? x : negative_slope_ * x;
            };
            return transform_tensor(activation_function, in_vol);
        }
    };

}
}
