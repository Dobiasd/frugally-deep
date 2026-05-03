// Copyright 2026, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

// Stateful GPT-2-style inference engine with a key/value cache, intended for
// the chat demo in examples/gpt2_chat. This bypasses the generic frugally-deep
// model runtime and computes the forward pass directly with Eigen so that
// successive ``step()`` calls reuse cached K/V tensors (cost ~constant per
// step rather than O(seq_len^2) for the full re-encode path).
//
// Reads the binary weights file produced by
// keras_export/save_gpt2_weights_bin.py.

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fdeep {
namespace llm {

    using ColMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowVector = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;

    namespace internal {

        constexpr std::int32_t GPT2_MAGIC = 0x47505432;
        constexpr std::int32_t GPT2_VERSION = 1;

        inline void read_exact(std::ifstream& in, void* dst, std::size_t bytes,
            const char* what)
        {
            in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
            if (!in || static_cast<std::size_t>(in.gcount()) != bytes) {
                throw std::runtime_error(std::string("short read: ") + what);
            }
        }

        struct gpt2_header {
            std::int32_t magic;
            std::int32_t version;
            std::int32_t num_layers;
            std::int32_t hidden_dim;
            std::int32_t num_heads;
            std::int32_t head_dim;
            std::int32_t intermediate_dim;
            std::int32_t vocab_size;
            std::int32_t max_position;
            std::int32_t reserved;
            float layer_norm_epsilon;
            char preset[64];
        };

    }  // namespace internal

    struct gpt2_block_weights {
        // Pre-attention LayerNorm.
        std::vector<float> attn_norm_gamma;
        std::vector<float> attn_norm_beta;
        // Attention projections (Q/K/V flattened to (hidden, hidden)).
        ColMatrix Wq, Wk, Wv;  // shape (hidden, hidden)
        std::vector<float> bq, bk, bv;  // length hidden
        // Output projection (hidden, hidden).
        ColMatrix Wo;
        std::vector<float> bo;
        // Pre-FFN LayerNorm.
        std::vector<float> ffn_norm_gamma;
        std::vector<float> ffn_norm_beta;
        // FFN.
        ColMatrix W1;  // (hidden, intermediate)
        std::vector<float> b1;
        ColMatrix W2;  // (intermediate, hidden)
        std::vector<float> b2;
    };

    class gpt2_cached_model {
    public:
        explicit gpt2_cached_model(const std::string& weights_path,
            std::size_t max_seq_len = 256)
            : max_seq_len_(max_seq_len)
        {
            std::ifstream in(weights_path, std::ios::binary);
            if (!in) {
                throw std::runtime_error("could not open " + weights_path);
            }
            internal::gpt2_header h{};
            internal::read_exact(in, &h, sizeof(h), "header");
            if (h.magic != internal::GPT2_MAGIC) {
                throw std::runtime_error("not a gpt2 weights file");
            }
            if (h.version != internal::GPT2_VERSION) {
                throw std::runtime_error("unsupported weights version");
            }
            num_layers_ = static_cast<std::size_t>(h.num_layers);
            hidden_dim_ = static_cast<std::size_t>(h.hidden_dim);
            num_heads_ = static_cast<std::size_t>(h.num_heads);
            head_dim_ = static_cast<std::size_t>(h.head_dim);
            intermediate_dim_ = static_cast<std::size_t>(h.intermediate_dim);
            vocab_size_ = static_cast<std::size_t>(h.vocab_size);
            max_position_ = static_cast<std::size_t>(h.max_position);
            layer_norm_epsilon_ = h.layer_norm_epsilon;
            if (max_seq_len_ > max_position_) {
                throw std::runtime_error("max_seq_len exceeds model's max_position");
            }
            if (num_heads_ * head_dim_ != hidden_dim_) {
                throw std::runtime_error("num_heads * head_dim != hidden_dim");
            }

            token_embedding_ = read_matrix(in, vocab_size_, hidden_dim_);
            position_embedding_ = read_matrix(in, max_position_, hidden_dim_);

            blocks_.resize(num_layers_);
            for (std::size_t i = 0; i < num_layers_; ++i) {
                auto& b = blocks_[i];
                b.attn_norm_gamma = read_vec(in, hidden_dim_);
                b.attn_norm_beta = read_vec(in, hidden_dim_);
                b.Wq = read_matrix(in, hidden_dim_, hidden_dim_);
                b.bq = read_vec(in, hidden_dim_);
                b.Wk = read_matrix(in, hidden_dim_, hidden_dim_);
                b.bk = read_vec(in, hidden_dim_);
                b.Wv = read_matrix(in, hidden_dim_, hidden_dim_);
                b.bv = read_vec(in, hidden_dim_);
                b.Wo = read_matrix(in, hidden_dim_, hidden_dim_);
                b.bo = read_vec(in, hidden_dim_);
                b.ffn_norm_gamma = read_vec(in, hidden_dim_);
                b.ffn_norm_beta = read_vec(in, hidden_dim_);
                b.W1 = read_matrix(in, hidden_dim_, intermediate_dim_);
                b.b1 = read_vec(in, intermediate_dim_);
                b.W2 = read_matrix(in, intermediate_dim_, hidden_dim_);
                b.b2 = read_vec(in, hidden_dim_);
            }

            final_norm_gamma_ = read_vec(in, hidden_dim_);
            final_norm_beta_ = read_vec(in, hidden_dim_);
            lm_head_ = read_matrix(in, hidden_dim_, vocab_size_);

            // Allocate KV cache: (num_layers, max_seq_len, hidden_dim) each.
            cache_k_.assign(num_layers_,
                ColMatrix::Zero(max_seq_len_, hidden_dim_));
            cache_v_.assign(num_layers_,
                ColMatrix::Zero(max_seq_len_, hidden_dim_));
            cur_len_ = 0;
        }

        std::size_t vocab_size() const { return vocab_size_; }
        std::size_t hidden_dim() const { return hidden_dim_; }
        std::size_t num_layers() const { return num_layers_; }
        std::size_t cur_len() const { return cur_len_; }
        std::size_t max_seq_len() const { return max_seq_len_; }

        void reset()
        {
            cur_len_ = 0;
        }

        // Process the entire prompt, building the cache. Returns logits for
        // predicting the token after the last prompt token.
        std::vector<float> prefill(const std::vector<int>& prompt_ids)
        {
            if (prompt_ids.empty()) {
                throw std::runtime_error("prompt is empty");
            }
            std::vector<float> logits;
            for (int id : prompt_ids) {
                logits = step(id);
            }
            return logits;
        }

        // Advance by one token. Returns logits for the next position.
        std::vector<float> step(int token_id)
        {
            if (cur_len_ >= max_seq_len_) {
                throw std::runtime_error("KV cache full");
            }
            if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size_) {
                throw std::runtime_error("token id out of range");
            }
            const std::size_t pos = cur_len_;
            // Embed: tok + pos.
            RowVector x = token_embedding_.row(token_id) + position_embedding_.row(pos);

            for (std::size_t i = 0; i < num_layers_; ++i) {
                forward_block(i, x, pos);
            }

            // Final norm.
            apply_layer_norm(x, final_norm_gamma_, final_norm_beta_);

            // LM head.
            RowVector logits = x * lm_head_;
            std::vector<float> out(vocab_size_);
            std::memcpy(out.data(), logits.data(), vocab_size_ * sizeof(float));

            // Commit step.
            ++cur_len_;
            return out;
        }

    private:
        static std::vector<float> read_vec(std::ifstream& in, std::size_t n)
        {
            std::vector<float> v(n);
            internal::read_exact(in, v.data(), n * sizeof(float), "vec");
            return v;
        }

        static ColMatrix read_matrix(std::ifstream& in, std::size_t rows, std::size_t cols)
        {
            ColMatrix m(rows, cols);
            internal::read_exact(in, m.data(), rows * cols * sizeof(float), "matrix");
            return m;
        }

        void apply_layer_norm(RowVector& x, const std::vector<float>& gamma,
            const std::vector<float>& beta) const
        {
            const float mean = x.mean();
            const float var = (x.array() - mean).square().mean();
            const float inv = 1.0f / std::sqrt(var + layer_norm_epsilon_);
            for (Eigen::Index j = 0; j < x.size(); ++j) {
                x(0, j) = ((x(0, j) - mean) * inv) * gamma[static_cast<std::size_t>(j)]
                    + beta[static_cast<std::size_t>(j)];
            }
        }

        static float gelu_approx(float v)
        {
            constexpr float c0 = 0.7978845608028654f;  // sqrt(2/pi)
            constexpr float c1 = 0.044715f;
            const float t = c0 * (v + c1 * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(t));
        }

        // One transformer block. Mutates ``x`` in place. ``pos`` is the
        // sequence position of the new token (also where the new K/V is
        // appended in the cache).
        void forward_block(std::size_t layer, RowVector& x, std::size_t pos)
        {
            const auto& b = blocks_[layer];

            // Pre-attention LN.
            RowVector n = x;
            apply_layer_norm(n, b.attn_norm_gamma, b.attn_norm_beta);

            // QKV projections (each (hidden,) = (hidden,) * (hidden,hidden)).
            RowVector q = n * b.Wq;
            RowVector k = n * b.Wk;
            RowVector v = n * b.Wv;
            for (std::size_t j = 0; j < hidden_dim_; ++j) {
                q(0, static_cast<Eigen::Index>(j)) += b.bq[j];
                k(0, static_cast<Eigen::Index>(j)) += b.bk[j];
                v(0, static_cast<Eigen::Index>(j)) += b.bv[j];
            }

            // Append K, V to cache at position `pos`.
            cache_k_[layer].row(static_cast<Eigen::Index>(pos)) = k;
            cache_v_[layer].row(static_cast<Eigen::Index>(pos)) = v;

            // Attention: for each head, compute scores against the cached
            // keys, softmax, weighted sum of cached values. Causal masking is
            // implicit — only positions [0..pos] have meaningful values.
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            const std::size_t cur = pos + 1;
            std::vector<float> attn_out(hidden_dim_, 0.0f);
            std::vector<float> scores(cur);
            for (std::size_t h = 0; h < num_heads_; ++h) {
                const std::size_t q_off = h * head_dim_;
                // Compute scores[t] = q_h . k_cache[t, h, :] * scale
                float max_score = -std::numeric_limits<float>::infinity();
                for (std::size_t t = 0; t < cur; ++t) {
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        dot += q(0, static_cast<Eigen::Index>(q_off + d))
                            * cache_k_[layer](static_cast<Eigen::Index>(t),
                                static_cast<Eigen::Index>(q_off + d));
                    }
                    const float s = dot * scale;
                    scores[t] = s;
                    if (s > max_score) max_score = s;
                }
                float total = 0.0f;
                for (std::size_t t = 0; t < cur; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    total += scores[t];
                }
                const float inv_total = 1.0f / total;
                for (std::size_t t = 0; t < cur; ++t) {
                    scores[t] *= inv_total;
                }
                // Weighted sum of values.
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    float s = 0.0f;
                    for (std::size_t t = 0; t < cur; ++t) {
                        s += scores[t]
                            * cache_v_[layer](static_cast<Eigen::Index>(t),
                                static_cast<Eigen::Index>(q_off + d));
                    }
                    attn_out[q_off + d] = s;
                }
            }
            // Output projection.
            Eigen::Map<RowVector> attn_vec(attn_out.data(), 1, static_cast<Eigen::Index>(hidden_dim_));
            RowVector attn_proj = attn_vec * b.Wo;
            for (std::size_t j = 0; j < hidden_dim_; ++j) {
                attn_proj(0, static_cast<Eigen::Index>(j)) += b.bo[j];
            }
            x += attn_proj;

            // Pre-FFN LN.
            n = x;
            apply_layer_norm(n, b.ffn_norm_gamma, b.ffn_norm_beta);

            // FFN: Dense(intermediate) -> GELU -> Dense(hidden).
            RowVector h1 = n * b.W1;
            for (std::size_t j = 0; j < intermediate_dim_; ++j) {
                h1(0, static_cast<Eigen::Index>(j))
                    = gelu_approx(h1(0, static_cast<Eigen::Index>(j)) + b.b1[j]);
            }
            RowVector h2 = h1 * b.W2;
            for (std::size_t j = 0; j < hidden_dim_; ++j) {
                h2(0, static_cast<Eigen::Index>(j)) += b.b2[j];
            }
            x += h2;
        }

        std::size_t num_layers_ = 0;
        std::size_t hidden_dim_ = 0;
        std::size_t num_heads_ = 0;
        std::size_t head_dim_ = 0;
        std::size_t intermediate_dim_ = 0;
        std::size_t vocab_size_ = 0;
        std::size_t max_position_ = 0;
        std::size_t max_seq_len_ = 0;
        float layer_norm_epsilon_ = 1e-5f;

        ColMatrix token_embedding_;     // (vocab, hidden)
        ColMatrix position_embedding_;  // (max_position, hidden)
        std::vector<gpt2_block_weights> blocks_;
        std::vector<float> final_norm_gamma_, final_norm_beta_;
        ColMatrix lm_head_;             // (hidden, vocab)

        std::vector<ColMatrix> cache_k_;  // num_layers x (max_seq_len, hidden)
        std::vector<ColMatrix> cache_v_;
        std::size_t cur_len_ = 0;
    };

}  // namespace llm
}  // namespace fdeep
