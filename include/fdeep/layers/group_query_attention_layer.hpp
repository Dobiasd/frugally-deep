// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class group_query_attention_layer : public layer {
    public:
        explicit group_query_attention_layer(const std::string& name,
            std::size_t head_dim, std::size_t num_query_heads, std::size_t num_kv_heads,
            bool use_bias, bool use_gate, const std::vector<tensor>& weights)
            : layer(name)
            , head_dim_(head_dim)
            , num_query_heads_(num_query_heads)
            , num_kv_heads_(num_kv_heads)
            , use_bias_(use_bias)
            , use_gate_(use_gate)
            , weights_(weights)
        {
            assertion(num_kv_heads_ > 0,
                "num_key_value_heads must be > 0.");
            assertion(num_query_heads_ % num_kv_heads_ == 0,
                "num_query_heads must be divisible by num_key_value_heads.");
        }

    protected:
        const std::size_t head_dim_;
        const std::size_t num_query_heads_;
        const std::size_t num_kv_heads_;
        const bool use_bias_;
        const bool use_gate_;
        const std::vector<tensor> weights_;

        // Weight order in Keras: Q, K, [Gate], V, Out, with bias right after each kernel.
        std::size_t weight_idx(std::size_t projection_idx) const
        {
            return projection_idx * (use_bias_ ? 2 : 1);
        }
        const tensor& q_kernel() const { return weights_[weight_idx(0)]; }
        const tensor& q_bias() const { return weights_[weight_idx(0) + 1]; }
        const tensor& k_kernel() const { return weights_[weight_idx(1)]; }
        const tensor& k_bias() const { return weights_[weight_idx(1) + 1]; }
        const tensor& gate_kernel() const { return weights_[weight_idx(2)]; }
        const tensor& gate_bias() const { return weights_[weight_idx(2) + 1]; }
        const tensor& v_kernel() const { return weights_[weight_idx(use_gate_ ? 3 : 2)]; }
        const tensor& v_bias() const { return weights_[weight_idx(use_gate_ ? 3 : 2) + 1]; }
        const tensor& o_kernel() const { return weights_[weight_idx(use_gate_ ? 4 : 3)]; }
        const tensor& o_bias() const { return weights_[weight_idx(use_gate_ ? 4 : 3) + 1]; }

        // Project an input of shape (T, dim) to (T, num_heads, head_dim) using
        // a kernel of shape (dim, num_heads, head_dim) and an optional bias of
        // shape (num_heads, head_dim).
        tensor project(const tensor& input,
            const tensor& kernel, const tensor* bias,
            std::size_t num_heads) const
        {
            const std::size_t T = input.shape().width_;
            const std::size_t in_dim = input.shape().depth_;
            const auto& xv = *input.as_vector();
            const auto& kv = *kernel.as_vector();
            const float_vec dummy_bias;
            const auto& bv = bias != nullptr ? *bias->as_vector() : dummy_bias;
            const bool has_bias = bias != nullptr;

            float_vec out(T * num_heads * head_dim_, 0);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t h = 0; h < num_heads; ++h) {
                    for (std::size_t c = 0; c < head_dim_; ++c) {
                        float_type acc = 0;
                        for (std::size_t d = 0; d < in_dim; ++d) {
                            acc += xv[t * in_dim + d]
                                * kv[d * num_heads * head_dim_ + h * head_dim_ + c];
                        }
                        if (has_bias)
                            acc += bv[h * head_dim_ + c];
                        out[t * num_heads * head_dim_ + h * head_dim_ + c] = acc;
                    }
                }
            }
            return tensor(tensor_shape(T, num_heads, head_dim_), std::move(out));
        }

        // Build the (T_q, T_k) softmax distribution for one query/kv-head pair.
        float_vec attention_distribution(
            const float_vec& Qv, const float_vec& Kv,
            std::size_t tq, std::size_t h_q,
            std::size_t T_k, std::size_t h_kv) const
        {
            const float_type inv_sqrt = static_cast<float_type>(1)
                / std::sqrt(static_cast<float_type>(head_dim_));
            float_vec scores(T_k, 0);
            float_type max_score = -std::numeric_limits<float_type>::infinity();
            for (std::size_t tk = 0; tk < T_k; ++tk) {
                float_type s = 0;
                for (std::size_t c = 0; c < head_dim_; ++c) {
                    s += Qv[tq * num_query_heads_ * head_dim_ + h_q * head_dim_ + c]
                        * Kv[tk * num_kv_heads_ * head_dim_ + h_kv * head_dim_ + c];
                }
                s *= inv_sqrt;
                scores[tk] = s;
                if (s > max_score)
                    max_score = s;
            }
            float_type sum_exp = 0;
            for (std::size_t tk = 0; tk < T_k; ++tk) {
                scores[tk] = std::exp(scores[tk] - max_score);
                sum_exp += scores[tk];
            }
            for (std::size_t tk = 0; tk < T_k; ++tk)
                scores[tk] /= sum_exp;
            return scores;
        }

        // Compute the attention output of shape (T_q, num_query_heads, head_dim).
        // Each query head h_q reads from kv head h_q / (num_query_heads / num_kv_heads).
        float_vec compute_attention(const tensor& Q, const tensor& K, const tensor& V) const
        {
            const std::size_t T_q = Q.shape().height_;
            const std::size_t T_k = K.shape().height_;
            const std::size_t group_size = num_query_heads_ / num_kv_heads_;
            const auto& Qv = *Q.as_vector();
            const auto& Kv = *K.as_vector();
            const auto& Vv = *V.as_vector();

            float_vec attn(T_q * num_query_heads_ * head_dim_, 0);
            for (std::size_t h_q = 0; h_q < num_query_heads_; ++h_q) {
                const std::size_t h_kv = h_q / group_size;
                for (std::size_t tq = 0; tq < T_q; ++tq) {
                    const auto distribution = attention_distribution(
                        Qv, Kv, tq, h_q, T_k, h_kv);
                    for (std::size_t c = 0; c < head_dim_; ++c) {
                        float_type acc = 0;
                        for (std::size_t tk = 0; tk < T_k; ++tk) {
                            acc += distribution[tk]
                                * Vv[tk * num_kv_heads_ * head_dim_ + h_kv * head_dim_ + c];
                        }
                        attn[tq * num_query_heads_ * head_dim_ + h_q * head_dim_ + c] = acc;
                    }
                }
            }
            return attn;
        }

        static void apply_sigmoid_gate(float_vec& attn, const float_vec& gate)
        {
            for (std::size_t i = 0; i < attn.size(); ++i) {
                const float_type sig = static_cast<float_type>(1)
                    / (static_cast<float_type>(1) + std::exp(-gate[i]));
                attn[i] *= sig;
            }
        }

        // Project (T_q, num_query_heads, head_dim) attn back to (T_q, out_dim).
        tensor output_projection(const float_vec& attn, std::size_t T_q) const
        {
            const tensor& out_kernel = o_kernel();
            const std::size_t out_dim = out_kernel.shape().depth_;
            const auto& Ov = *out_kernel.as_vector();
            const float_vec dummy_bias;
            const auto& obv = use_bias_ ? *o_bias().as_vector() : dummy_bias;

            float_vec result(T_q * out_dim, 0);
            for (std::size_t t = 0; t < T_q; ++t) {
                for (std::size_t d = 0; d < out_dim; ++d) {
                    float_type acc = 0;
                    for (std::size_t h = 0; h < num_query_heads_; ++h) {
                        for (std::size_t c = 0; c < head_dim_; ++c) {
                            acc += attn[t * num_query_heads_ * head_dim_ + h * head_dim_ + c]
                                * Ov[h * head_dim_ * out_dim + c * out_dim + d];
                        }
                    }
                    if (use_bias_)
                        acc += obv[d];
                    result[t * out_dim + d] = acc;
                }
            }
            return tensor(tensor_shape(T_q, out_dim), std::move(result));
        }

        tensors apply_impl(const tensors& input) const override
        {
            assertion(input.size() == 2 || input.size() == 3,
                "GroupQueryAttention requires 2 or 3 inputs (query, value[, key]).");
            const tensor& query_raw = input[0];
            const tensor& value_raw = input[1];
            const tensor& key_raw = input.size() > 2 ? input[2] : value_raw;
            assertion(query_raw.shape().rank() == 2
                    && value_raw.shape().rank() == 2 && key_raw.shape().rank() == 2,
                "GroupQueryAttention expects rank-2 inputs (T, dim).");

            const tensor* qb = use_bias_ ? &q_bias() : nullptr;
            const tensor* kb = use_bias_ ? &k_bias() : nullptr;
            const tensor* vb = use_bias_ ? &v_bias() : nullptr;

            const tensor Q = project(query_raw, q_kernel(), qb, num_query_heads_);
            const tensor K = project(key_raw, k_kernel(), kb, num_kv_heads_);
            const tensor V = project(value_raw, v_kernel(), vb, num_kv_heads_);

            float_vec attn = compute_attention(Q, K, V);

            if (use_gate_) {
                const tensor* gb = use_bias_ ? &gate_bias() : nullptr;
                const tensor gate = project(query_raw, gate_kernel(), gb, num_query_heads_);
                apply_sigmoid_gate(attn, *gate.as_vector());
            }

            return { output_projection(attn, Q.shape().height_) };
        }
    };

}
}
