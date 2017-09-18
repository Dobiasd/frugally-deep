// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// Abstract base class for actication layers
// https://en.wikipedia.org/wiki/Activation_function
class activation_layer : public layer
{
public:
    explicit activation_layer(const std::string& name) :
        layer(name)
    {
    }
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {transform_input(input)};
    }

protected:
    virtual matrix3d transform_input(const matrix3d& input) const = 0;
};

inline matrix3ds apply_activation_layer(const activation_layer_ptr& ptr,
    const matrix3ds& input)
{
    return ptr == nullptr ? input : ptr->apply(input);
}

} // namespace fd
