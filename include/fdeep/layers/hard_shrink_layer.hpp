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

    class hard_shrink_layer : public activation_layer {
    public:
        explicit hard_shrink_layer(const std::string& name, float_type threshold)
            : activation_layer(name)
            , threshold_(threshold)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return transform_tensor([this](float_type x) {
                return std::abs(x) > threshold_ ? x : 0;
            },
                in_vol);
        }
        float_type threshold_;
    };

}
}
