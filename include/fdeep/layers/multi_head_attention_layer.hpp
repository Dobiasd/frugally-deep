// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/softmax_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class multi_head_attention_layer : public layer
{
public:
    explicit multi_head_attention_layer(const std::string& name)
        : layer(name)
    {
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2 or input.size() == 3, "Invalid number of inputs for MultiHeadAttention layer.");
        // todo: implement
        return input;
    }
};

} } // namespace fdeep, namespace internal
