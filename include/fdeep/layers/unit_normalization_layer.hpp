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

    class unit_normalization_layer : public layer {
    public:
        explicit unit_normalization_layer(const std::string& name,
            std::vector<int> axes)
            : layer(name)
            , axes_(axes)
        {
        }

    protected:
        std::vector<int> axes_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            return { l2_normalize(input, axes_) };
        }
    };

}
}
