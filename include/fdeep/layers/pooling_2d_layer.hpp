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
        const shape2& pool_size, const shape2& strides, padding p,
        bool padding_valid_uses_offset, bool padding_same_uses_offset) :
        layer(name),
        pool_size_(pool_size),
        strides_(strides),
        padding_(p),
        padding_valid_uses_offset_(padding_valid_uses_offset),
        padding_same_uses_offset_(padding_same_uses_offset)
    {
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {pool(input)};
    }

    bool use_offset() const
    {
        return
            (padding_ == padding::valid && padding_valid_uses_offset_) ||
            (padding_ == padding::same && padding_same_uses_offset_);
    }

    virtual tensor5 pool(const tensor5& input) const = 0;

    shape2 pool_size_;
    shape2 strides_;
    padding padding_;
    bool padding_valid_uses_offset_;
    bool padding_same_uses_offset_;
};

} } // namespace fdeep, namespace internal
