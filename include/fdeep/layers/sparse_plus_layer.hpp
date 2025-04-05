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

    class sparse_plus_layer : public activation_layer {
    public:
        explicit sparse_plus_layer(const std::string& name)
            : activation_layer(name)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return transform_tensor([this](float_type x) {
                if (x <= -1) {
                    return static_cast<float_type>(0);
                } else if (x < 1) {
                    return static_cast<float_type>(0.25) * (x + static_cast<float_type>(1)) * (x + static_cast<float_type>(1));
                }
                return x;
            },
                in_vol);
        }
    };

}
}
