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
    matrix3d apply(const matrix3d& input) const override
    {
        return transform_input(input);
    }

protected:
    virtual matrix3d transform_input(const matrix3d& input) const = 0;
};

} // namespace fd
