// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <limits>
#include <string>

namespace fdeep {
namespace internal {

    class gelu_layer : public activation_layer {
    public:
        explicit gelu_layer(const std::string& name, bool approximate = false)
            : activation_layer(name)
            , approximate_(approximate)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            return approximate_
                ? transform_tensor(gelu_approximate_activation, in_vol)
                : transform_tensor(gelu_activation, in_vol);
        }
        bool approximate_;
    };

}
}
