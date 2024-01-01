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

    class add_layer : public layer {
    public:
        explicit add_layer(const std::string& name)
            : layer(name)
        {
        }

    protected:
        tensors apply_impl(const tensors& input) const override
        {
            return { sum_tensors(input) };
        }
    };

}
}
