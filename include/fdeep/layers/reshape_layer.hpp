// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class reshape_layer : public layer
{
public:
    explicit reshape_layer(const std::string& name,
        const tensor_shape& target_shape)
        : layer(name),
        target_shape_(target_shape)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        return {tensor(target_shape_, input.as_vector())};
    }
    tensor_shape target_shape_;
};

} } // namespace fdeep, namespace internal
