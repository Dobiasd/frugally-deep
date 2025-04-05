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

    class log_sigmoid_layer : public activation_layer {
    public:
        explicit log_sigmoid_layer(const std::string& name)
            : activation_layer(name)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return transform_tensor([this](float_type x) {
                return std::log(1 / (1 + std::exp(-x)));
            },
                in_vol);
        }
    };

}
}
