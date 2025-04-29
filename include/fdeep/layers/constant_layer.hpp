// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep {
namespace internal {

    class constant_layer : public layer {
    public:
        explicit constant_layer(const std::string& name, const tensor& constant_tensor)
            : layer(name)
            , constant_tensor_(constant_tensor)
        {
        }

    protected:
        tensors apply_impl(const tensors&) const override
        {
            return { constant_tensor_ };
        }
        tensor constant_tensor_;
    };

}
}