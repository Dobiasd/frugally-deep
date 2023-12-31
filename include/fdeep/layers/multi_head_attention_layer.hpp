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
            query_dense_(create_dense_layers(weights_and_biases, use_bias, num_heads, 0, name + "_query_dense")),
            value_dense_(create_dense_layers(weights_and_biases, use_bias, num_heads, 2, name + "_value_dense")),
            key_dense_(create_dense_layers(weights_and_biases, use_bias, num_heads, 1, name + "_key_dense")),
            output_dense_(create_dense_layers(weights_and_biases, use_bias, num_heads, 3, name + "_output_dense"))
    {
    }
private:
    std::vector<dense_layer> create_dense_layers(
        const tensors& weights_and_biases, bool use_bias, const std::size_t num_heads,
        const std::size_t index, const std::string& name)
    {
        const std::size_t index_factor = use_bias ? 2 : 1;
        const tensor weights = weights_and_biases[index_factor * index];
        const std::size_t units = weights.shape().depth_;
        const tensor biases = use_bias ?
            weights_and_biases[index_factor * index + 1] :
            tensor(index == 3 ? tensor_shape(num_heads, 1, units) : tensor_shape(num_heads, units), 0);
        const auto weights_per_head =
            index == 3 ? tensor_to_tensors_height_slices(weights) : tensor_to_tensors_width_slices(weights);
        const auto biases_per_head =
            index == 3 ? tensor_to_tensors_height_slices(biases) : tensor_to_tensors_width_slices(biases);
        assertion(weights_per_head.size() == num_heads, "Invalid weights for number of heads.");
        assertion(biases_per_head.size() == num_heads, "Invalid biases for number of heads.");
        const std::vector<dense_layer> dense_layers = 
            fplus::transform(
                [&](const std::pair<std::size_t, std::pair<tensor, tensor>>& n_and_w_with_b)
                {
                    return dense_layer(
                        name + "_" + std::to_string(n_and_w_with_b.first),
                        units,
                        *n_and_w_with_b.second.first.as_vector(),
                        *n_and_w_with_b.second.second.as_vector());
                },
                fplus::enumerate(fplus::zip(weights_per_head, biases_per_head)));
        return dense_layers;
    }
    tensors extract_biases(const tensors& saved_weights, bool use_bias)
    {
        return use_bias ? fplus::unweave(saved_weights).second : tensors();
    }
    tensor apply_head(
        const tensor& query_raw,
        const tensor& value_raw,
        const tensor& key_raw,
        std::size_t head_index) const
    {
        assertion(
            query_raw.shape().rank() == 2 &&
            value_raw.shape().rank() == 2 &&
            key_raw.shape().rank() == 2 &&
            query_raw.shape().depth_ == value_raw.shape().depth_ &&
            query_raw.shape().depth_ == key_raw.shape().depth_ &&
            value_raw.shape().width_ == key_raw.shape().width_,
            "Invalid shapes; need a query tensor of shape (B, T, dim) and a value/key tensor of shape (B, S, dim)."
        );
        const tensor query = query_dense_[head_index].apply({query_raw}).front();
        const tensor value = value_dense_[head_index].apply({value_raw}).front();
        const tensor key = key_dense_[head_index].apply({key_raw}).front();

        // https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        // https://dmol.pub/dl/attention.html#multi-head-attention-block
        // https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/attention/multi_head_attention.py
        // https://gist.github.com/sevagh/b71d253a347a9b59c026580625452fc5
        const tensor scores = dot_product_tensors(query, transpose(key), std::vector<int>({2, 1}), false);
        const tensor distribution = softmax(scores);
        const tensor output = dot_product_tensors(distribution, value, std::vector<int>({2, 1}), false);
        return output_dense_[head_index].apply({output}).front(); // todo
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2 || input.size() == 3, "Invalid number of inputs for MultiHeadAttention layer.");
        const tensor query_raw = input[0];
        const tensor value_raw = input[1];
        const tensor key_raw = input.size() > 2 ? input[2] : value_raw;
        return {apply_head(query_raw, value_raw, key_raw, 0)}; // todo: all
    }
    std::size_t num_heads_;
    std::size_t key_dim_;
    std::size_t value_dim_;
    std::vector<std::size_t> attention_axes_;
    std::vector<dense_layer> query_dense_;
    std::vector<dense_layer> value_dense_;
    std::vector<dense_layer> key_dense_;
    std::vector<dense_layer> output_dense_;
};

} } // namespace fdeep, namespace internal
