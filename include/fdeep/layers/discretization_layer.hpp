// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class discretization_layer : public layer {
    public:
        explicit discretization_layer(const std::string& name,
            const std::vector<float_type>& boundaries)
            : layer(name)
            , boundaries_(boundaries)
        {
        }

    protected:
        std::vector<float_type> boundaries_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto& src = *input.as_vector();
            float_vec out(src.size(), 0);
            for (std::size_t i = 0; i < src.size(); ++i) {
                const auto it = std::upper_bound(
                    boundaries_.begin(), boundaries_.end(), src[i]);
                out[i] = static_cast<float_type>(it - boundaries_.begin());
            }
            return { tensor(input.shape(), std::move(out)) };
        }
    };

}
}
