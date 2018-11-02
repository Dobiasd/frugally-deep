// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

// Abstract base class for global pooling layers
class global_pooling_2d_layer : public layer
{
public:
    explicit global_pooling_2d_layer(const std::string& name) :
        layer(name)
    {
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {pool(input)};
    }
    virtual tensor5 pool(const tensor5& input) const = 0;
};

} } // namespace fdeep, namespace internal
