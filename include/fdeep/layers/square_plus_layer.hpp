// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <algorithm>
#include <string>

namespace fdeep {
namespace internal {

    class square_plus_layer : public activation_layer {
    public:
        explicit square_plus_layer(const std::string& name, float_type b)
            : activation_layer(name)
            , b_(b)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return transform_tensor([this](float_type x) {
                return (x + std::sqrt(x * x + b_)) / static_cast<float_type>(2.0);
            },
                in_vol);
        }
        float_type b_;
    };

}
}
