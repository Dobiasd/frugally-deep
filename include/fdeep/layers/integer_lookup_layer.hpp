// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fdeep {
namespace internal {

    class integer_lookup_layer : public layer {
    public:
        explicit integer_lookup_layer(const std::string& name,
            const std::vector<std::int64_t>& vocabulary,
            std::size_t num_oov_indices,
            bool has_mask_token,
            std::int64_t mask_token)
            : layer(name)
            , has_mask_token_(has_mask_token)
            , mask_token_(mask_token)
            , num_oov_indices_(num_oov_indices)
            , vocab_offset_((has_mask_token ? 1 : 0) + num_oov_indices)
            , vocab_index_()
        {
            // Keras hashes OOV inputs across multiple OOV buckets via FarmHash,
            // which fdeep does not currently replicate. Restrict to a single
            // OOV bucket (the common case) to keep behavior bit-identical.
            assertion(num_oov_indices_ <= 1,
                "IntegerLookup with num_oov_indices > 1 is not supported.");
            for (std::size_t i = 0; i < vocabulary.size(); ++i)
                vocab_index_[vocabulary[i]] = i;
        }

    protected:
        const bool has_mask_token_;
        const std::int64_t mask_token_;
        const std::size_t num_oov_indices_;
        const std::size_t vocab_offset_;
        std::unordered_map<std::int64_t, std::size_t> vocab_index_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto& src = *input.as_vector();
            float_vec out(src.size(), 0);
            for (std::size_t i = 0; i < src.size(); ++i) {
                const std::int64_t v = static_cast<std::int64_t>(src[i]);
                if (has_mask_token_ && v == mask_token_) {
                    out[i] = 0;
                    continue;
                }
                const auto it = vocab_index_.find(v);
                if (it != vocab_index_.end()) {
                    out[i] = static_cast<float_type>(vocab_offset_ + it->second);
                } else {
                    // num_oov_indices_ <= 1 (asserted in ctor): OOV maps to 0,
                    // shifted by 1 if a mask token occupies index 0.
                    out[i] = static_cast<float_type>(has_mask_token_ ? 1 : 0);
                }
            }
            return { tensor(input.shape(), std::move(out)) };
        }
    };

}
}
