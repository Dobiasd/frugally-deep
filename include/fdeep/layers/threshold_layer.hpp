// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <algorithm>
#include <string>

namespace fdeep {
namespace internal {

    class threshold_layer : public activation_layer {
    public:
        explicit threshold_layer(const std::string& name, float_type threshold, float_type default_value)
            : activation_layer(name)
            , threshold_(threshold)
            , default_value_(default_value)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return transform_tensor([this](float_type x) {
                return x > threshold_ ? x : default_value_;
            },
                in_vol);
        }
        float_type threshold_;
        float_type default_value_;
    };

}
}
