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
    explicit multi_head_attention_layer(const std::string& name,
        std::size_t num_heads, std::size_t key_dim, std::size_t value_dim, 
        bool use_bias, const std::vector<std::size_t>& attention_axes,
        const std::vector<float_vec>& weights)
        : layer(name), num_heads_(num_heads), key_dim_(key_dim),
            value_dim_(value_dim), use_bias_(use_bias), attention_axes_(attention_axes),
            weights_(weights)
    {
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2 || input.size() == 3, "Invalid number of inputs for MultiHeadAttention layer.");
        //const tensor& query = input[0];
        //const tensor& value = input[1];
        //const tensor& key = input.size() > 2 ? input[2] : value;
        // https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/attention/multi_head_attention.py
        return input;
    }
    std::size_t num_heads_;
    std::size_t key_dim_;
    std::size_t value_dim_;
    bool use_bias_;
    std::vector<std::size_t> attention_axes_;
    std::vector<float_vec> weights_;
};

} } // namespace fdeep, namespace internal
