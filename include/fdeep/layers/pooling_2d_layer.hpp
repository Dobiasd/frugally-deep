// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/convolution.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

// Abstract base class for pooling layers
class pooling_2d_layer : public layer
{
public:
    explicit pooling_2d_layer(const std::string& name,
        const shape2& pool_size, const shape2& strides, bool channels_first, padding p) :
        layer(name),
        pool_size_(pool_size),
        strides_(strides),
        channels_first_(channels_first),
        padding_(p)
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

    shape2 pool_size_;
    shape2 strides_;
    bool channels_first_;
    padding padding_;
};

} } // namespace fdeep, namespace internal
