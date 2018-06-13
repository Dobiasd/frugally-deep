// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
#include <string>

namespace fdeep { namespace internal
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
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        const auto f = [this](const tensor3& t) -> tensor3
        {
            return transform_input(t);
        };
        return fplus::transform(f, inputs);
    }

protected:
    virtual tensor3 transform_input(const tensor3& input) const = 0;
};

inline tensor3s apply_activation_layer(
    const activation_layer_ptr& ptr,
    const tensor3s& input)
{
    return ptr == nullptr ? input : ptr->apply(input);
}

} } // namespace fdeep, namespace internal
