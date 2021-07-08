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
        const tensor_shape_variable& target_shape)
        : layer(name),
        target_shape_(target_shape)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        const auto fixed_target_shape = derive_fixed_tensor_shape(
            input.shape().volume(), target_shape_);
        return {tensor(fixed_target_shape, input.as_vector())};
    }
    tensor_shape_variable target_shape_;
};

} } // namespace fdeep, namespace internal
