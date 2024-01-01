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

    class concatenate_layer : public layer {
    public:
        explicit concatenate_layer(const std::string& name, int axis)
            : layer(name)
            , axis_(axis)
        {
        }

    protected:
        tensors apply_impl(const tensors& input) const override
        {
            return { concatenate_tensors(input, axis_) };
        }
        int axis_;
    };

}
}
