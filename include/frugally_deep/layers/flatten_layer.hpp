// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.hpp"

namespace fdeep { namespace internal
{

// Converts a volume into single column volume (shape3(n, 1, 1)).
class flatten_layer : public layer
{
public:
    explicit flatten_layer(const std::string& name) :
            layer(name)
    {
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {flatten_tensor3(input)};
    }
};

} } // namespace fdeep, namespace internal
