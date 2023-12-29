// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/dense_layer.hpp"
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
        const std::vector<tensor>& weights_and_biases)
        : layer(name), num_heads_(num_heads), key_dim_(key_dim),
            value_dim_(value_dim), attention_axes_(attention_axes),
            query_dense_(create_dense_layer(weights_and_biases, use_bias, 0, name + "_query_dense")),
            value_dense_(create_dense_layer(weights_and_biases, use_bias, 1, name + "_value_dense")),
            key_dense_(create_dense_layer(weights_and_biases, use_bias, 2, name + "_key_dense")),
            output_dense_(create_dense_layer(weights_and_biases, use_bias, 3, name + "_output_dense"))
    {
    }
private:
    dense_layer create_dense_layer(
        const tensors& weights_and_biases, bool use_bias,
        std::size_t index, const std::string& name)
    {
        const std::size_t index_factor = use_bias ? 2 : 1;
        const tensor weights = weights_and_biases[index_factor * index];
        const std::size_t n = weights.shape().width_ * weights.shape().depth_;
        const tensor biases = use_bias ?
            weights_and_biases[index_factor * index + 1] :
            tensor(tensor_shape(n), 1);
        return dense_layer(name, n, *weights.as_vector(), *biases.as_vector());
    }
    tensors extract_biases(const tensors& saved_weights, bool use_bias)
    {
        return use_bias ? fplus::unweave(saved_weights).second : tensors();
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2 || input.size() == 3, "Invalid number of inputs for MultiHeadAttention layer.");
        const tensor query_raw = input[0];
        const tensor value_raw = input[1];
        const tensor key_raw = input.size() > 2 ? input[2] : value_raw;
        const tensor query = query_dense_.apply({query_raw}).front();
        const tensor value = value_dense_.apply({value_raw}).front();
        const tensor key = key_dense_.apply({key_raw}).front();
        assertion(
            query.shape().rank() == 2 &&
            value.shape().rank() == 2 &&
            key.shape().rank() == 2 &&
            query.shape().depth_ == value.shape().depth_ &&
            query.shape().depth_ == key.shape().depth_ &&
            value.shape().width_ == key.shape().width_,
            "Invalid shapes; need a query tensor of shape (B, T, dim) and a value/key tensor of shape (B, S, dim)."
        );
        // https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        // https://dmol.pub/dl/attention.html#multi-head-attention-block
        // https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/attention/multi_head_attention.py
        // https://gist.github.com/sevagh/b71d253a347a9b59c026580625452fc5
        const tensor scores = dot_product_tensors(query, transpose(key), std::vector<int>({2, 1}), false);
        const tensor distribution = softmax(scores);
        const tensor output = dot_product_tensors(distribution, value, std::vector<int>({2, 1}), false);
        return output_dense_.apply({output});
    }
    std::size_t num_heads_;
    std::size_t key_dim_;
    std::size_t value_dim_;
    std::vector<std::size_t> attention_axes_;
    dense_layer query_dense_;
    dense_layer value_dense_;
    dense_layer key_dense_;
    dense_layer output_dense_;
};

} } // namespace fdeep, namespace internal
