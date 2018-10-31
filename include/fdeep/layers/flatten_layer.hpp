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

// Converts a volume into single column volume (shape5(1, 1, 1, 1, n)).
class flatten_layer : public layer
{
public:
    explicit flatten_layer(const std::string& name) :
            layer(name)
    {
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {flatten_tensor5(input)};
    }
};

} } // namespace fdeep, namespace internal
