// Copyright 2026, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

// Autoregressive token generation on top of a frugally-deep model exported
// from a GPT-2-style backbone with a tied LM head. The model is expected to
// take two integer inputs (token_ids, position_ids) of shape (seq_len,) and
// produce vocabulary logits of shape (seq_len, vocab_size).
//
// This implementation does NOT use a key/value cache — every step re-runs the
// full forward pass over the (prompt + generated-so-far) tokens, padded out
// to the model's fixed sequence length. For long contexts this is slow.

#pragma once

#include <fdeep/fdeep.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

namespace fdeep {
namespace llm {

    struct gpt2_generation_params {
        std::size_t max_new_tokens = 32;
        // Temperature 0 means greedy/argmax. Otherwise scales logits.
        float temperature = 0.0f;
        // 0 disables top-k filtering.
        std::size_t top_k = 0;
        // Optional seed for the sampling RNG.
        std::uint64_t seed = 0;
        // Optional callback invoked with each newly generated token id, just
        // after it is appended. Useful for streaming output. Returning ``false``
        // stops generation.
        std::function<bool(int)> on_token = {};
    };

    class gpt2_generator {
    public:
        gpt2_generator(const fdeep::model& model_, std::size_t seq_len,
            std::size_t vocab_size, int eos_token_id, std::size_t pad_token_id = 0)
            : model_(model_)
            , seq_len_(seq_len)
            , vocab_size_(vocab_size)
            , eos_token_id_(eos_token_id)
            , pad_token_id_(pad_token_id)
        {
            if (seq_len_ == 0) {
                throw std::runtime_error("seq_len must be positive");
            }
        }

        std::vector<int> generate(const std::vector<int>& prompt_ids,
            const gpt2_generation_params& params) const
        {
            if (prompt_ids.empty()) {
                throw std::runtime_error("prompt_ids must not be empty");
            }
            if (prompt_ids.size() >= seq_len_) {
                throw std::runtime_error(
                    "prompt is at least as long as the model's seq_len; "
                    "no room to generate");
            }

            std::mt19937_64 rng(params.seed);

            // Pre-fill the fixed-length token / position buffers.
            fdeep::float_vec tokens(seq_len_, static_cast<float>(pad_token_id_));
            fdeep::float_vec positions(seq_len_);
            for (std::size_t i = 0; i < seq_len_; ++i) {
                positions[i] = static_cast<float>(i);
            }
            for (std::size_t i = 0; i < prompt_ids.size(); ++i) {
                tokens[i] = static_cast<float>(prompt_ids[i]);
            }

            std::vector<int> generated;
            generated.reserve(params.max_new_tokens);

            for (std::size_t step = 0; step < params.max_new_tokens; ++step) {
                const std::size_t cur_len = prompt_ids.size() + step;
                if (cur_len >= seq_len_) {
                    break;  // out of room
                }

                fdeep::float_vec tokens_copy = tokens;
                fdeep::float_vec positions_copy = positions;
                const fdeep::tensor token_t(fdeep::tensor_shape(seq_len_),
                    std::move(tokens_copy));
                const fdeep::tensor position_t(fdeep::tensor_shape(seq_len_),
                    std::move(positions_copy));
                const auto out = model_.predict({ token_t, position_t });
                if (out.empty()) {
                    throw std::runtime_error("model produced no output");
                }
                const auto& logits = *out.front().as_vector();
                if (logits.size() != seq_len_ * vocab_size_) {
                    throw std::runtime_error(
                        "unexpected logit count: got "
                        + std::to_string(logits.size()) + ", expected "
                        + std::to_string(seq_len_ * vocab_size_));
                }

                // Read logits at position (cur_len - 1), the last filled slot.
                const std::size_t row = cur_len - 1;
                const float* row_ptr = logits.data() + row * vocab_size_;

                int next_id = sample_next(row_ptr, params, rng);
                generated.push_back(next_id);
                tokens[cur_len] = static_cast<float>(next_id);

                if (params.on_token && !params.on_token(next_id)) {
                    break;
                }
                if (next_id == eos_token_id_) {
                    break;
                }
            }
            return generated;
        }

    private:
        int sample_next(const float* logits, const gpt2_generation_params& p,
            std::mt19937_64& rng) const
        {
            if (p.temperature <= 0.0f) {
                // Greedy.
                std::size_t best = 0;
                float best_val = logits[0];
                for (std::size_t i = 1; i < vocab_size_; ++i) {
                    if (logits[i] > best_val) {
                        best_val = logits[i];
                        best = i;
                    }
                }
                return static_cast<int>(best);
            }

            // Build a vector of (logit, idx). For top-k, keep only the top k.
            std::vector<std::pair<float, int>> candidates;
            candidates.reserve(vocab_size_);
            for (std::size_t i = 0; i < vocab_size_; ++i) {
                candidates.emplace_back(logits[i], static_cast<int>(i));
            }
            const float inv_temp = 1.0f / p.temperature;
            for (auto& c : candidates) c.first *= inv_temp;

            if (p.top_k > 0 && p.top_k < candidates.size()) {
                std::partial_sort(candidates.begin(),
                    candidates.begin() + static_cast<std::ptrdiff_t>(p.top_k),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                candidates.resize(p.top_k);
            }

            float m = candidates[0].first;
            for (const auto& c : candidates) m = std::max(m, c.first);
            float total = 0.0f;
            for (auto& c : candidates) {
                c.first = std::exp(c.first - m);
                total += c.first;
            }
            std::uniform_real_distribution<float> dist(0.0f, total);
            const float pick = dist(rng);
            float acc = 0.0f;
            for (const auto& c : candidates) {
                acc += c.first;
                if (acc >= pick) {
                    return c.second;
                }
            }
            return candidates.back().second;
        }

        const fdeep::model& model_;
        std::size_t seq_len_;
        std::size_t vocab_size_;
        int eos_token_id_;
        std::size_t pad_token_id_;
    };

}  // namespace llm
}  // namespace fdeep
