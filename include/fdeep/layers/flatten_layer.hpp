// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

// Converts a volume into single column volume (tensor_shape(n)).
class flatten_layer : public layer
{
public:
    explicit flatten_layer(const std::string& name) :
            layer(name)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        return {flatten_tensor(input)};
    }
};

} } // namespace fdeep, namespace internal
