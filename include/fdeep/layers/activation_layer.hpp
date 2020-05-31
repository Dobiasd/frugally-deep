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
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto f = [this](const tensor& t) -> tensor
        {
            return transform_input(t);
        };
        return fplus::transform(f, inputs);
    }

protected:
    virtual tensor transform_input(const tensor& input) const = 0;
};

inline tensors apply_activation_layer(
    const activation_layer_ptr& ptr,
    const tensors& input)
{
    return ptr == nullptr ? input : ptr->apply(input);
}

} } // namespace fdeep, namespace internal
