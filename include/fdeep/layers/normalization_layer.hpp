// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class normalization_layer : public layer
{
public:
    explicit normalization_layer(
            const std::string& name,
            int axis,
            const float_vec& mean, const float_vec& variance)
        : layer(name),
        axis_(axis),
        mean_(mean),
        variance_(variance)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        return {input}; // todo: Implement the actual normalization.
    }
    int axis_;
    float_vec mean_;
    float_vec variance_;
};

} } // namespace fdeep, namespace internal
