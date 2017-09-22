// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

// Converts a volume into single column volume (size3d(n, 1, 1)).
class flatten_layer : public layer
{
public:
    explicit flatten_layer(const std::string& name) :
            layer(name)
    {
    }
protected:
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {flatten_matrix3d(input)};
    }
};

} // namespace fd
