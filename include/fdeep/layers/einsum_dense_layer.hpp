// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <map>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class einsum_dense_layer : public layer {
    public:
        explicit einsum_dense_layer(const std::string& name,
            const std::string& equation,
            const std::vector<int>& full_output_shape,
            const std::string& bias_axes,
            const tensor& kernel,
            const tensor& bias)
            : layer(name)
            , equation_(equation)
            , full_output_shape_(full_output_shape)
            , bias_axes_(bias_axes)
            , kernel_(kernel)
            , bias_(bias)
            , lhs_(parse_lhs(equation))
            , rhs_kernel_(parse_rhs_kernel(equation))
            , rhs_(parse_rhs(equation))
            , summed_(compute_summed(lhs_, rhs_kernel_, rhs_))
        {
        }

    protected:
        const std::string equation_;
        const std::vector<int> full_output_shape_;
        const std::string bias_axes_;
        const tensor kernel_;
        const tensor bias_;

        const std::string lhs_;
        const std::string rhs_kernel_;
        const std::string rhs_;
        const std::string summed_;

        // Parses "lhs,rhs_kernel->rhs" into its three pieces.
        static std::string parse_lhs(const std::string& eq)
        {
            return eq.substr(0, eq.find(','));
        }

        static std::string parse_rhs_kernel(const std::string& eq)
        {
            const auto comma = eq.find(',');
            return eq.substr(comma + 1, eq.find("->") - comma - 1);
        }

        static std::string parse_rhs(const std::string& eq)
        {
            return eq.substr(eq.find("->") + 2);
        }

        // Chars that appear on both sides of the equation but not in the
        // output are summed away.
        static std::string compute_summed(const std::string& lhs,
            const std::string& rhs_kernel, const std::string& rhs)
        {
            std::string summed;
            for (char c : lhs)
                if (rhs.find(c) == std::string::npos
                    && rhs_kernel.find(c) != std::string::npos)
                    summed.push_back(c);
            return summed;
        }

        // Row-major strides for a list of dim sizes.
        static std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& sizes)
        {
            std::vector<std::size_t> strides(sizes.size());
            std::size_t s = 1;
            for (std::size_t i = sizes.size(); i-- > 0;) {
                strides[i] = s;
                s *= sizes[i];
            }
            return strides;
        }

        static std::size_t product(const std::vector<std::size_t>& v)
        {
            std::size_t p = 1;
            for (auto x : v)
                p *= x;
            return p;
        }

        // Decodes a flat index back into per-char coordinates and writes them
        // into pos. Used to enumerate output positions and summed-axis combos.
        static void decode_index(std::size_t idx, const std::string& chars,
            const std::vector<std::size_t>& strides,
            std::map<char, std::size_t>& pos)
        {
            for (std::size_t i = 0; i < chars.size(); ++i) {
                pos[chars[i]] = idx / strides[i];
                idx %= strides[i];
            }
        }

        static std::size_t encode_offset(const std::string& chars,
            const std::vector<std::size_t>& strides,
            const std::map<char, std::size_t>& pos)
        {
            std::size_t off = 0;
            for (std::size_t i = 0; i < chars.size(); ++i)
                off += pos.at(chars[i]) * strides[i];
            return off;
        }

        // Build a {char -> dim size} map by combining info from kernel
        // dimensions, the input tensor, and (as a fallback) full_output_shape_.
        std::map<char, std::size_t> derive_char_sizes(const tensor& input) const
        {
            std::map<char, std::size_t> char_size;

            // Kernel: rhs_kernel_ chars are the trailing dims of the kernel tensor.
            const auto kernel_dims = kernel_.shape().dimensions();
            assertion(rhs_kernel_.size() <= kernel_dims.size(),
                "EinsumDense: kernel chars exceed kernel dimensions.");
            const std::size_t k_offset = kernel_dims.size() - rhs_kernel_.size();
            for (std::size_t i = 0; i < rhs_kernel_.size(); ++i)
                char_size[rhs_kernel_[i]] = kernel_dims[k_offset + i];

            // Input: lhs chars beyond the input rank are dropped (None) batch
            // dims with size 1; trailing chars map onto physical dims.
            const auto input_dims = input.shape().dimensions();
            const std::size_t input_rank = input.shape().rank();
            assertion(lhs_.size() >= input_rank,
                "EinsumDense: input rank exceeds lhs char count.");
            const std::size_t leading = lhs_.size() - input_rank;
            for (std::size_t i = 0; i < lhs_.size(); ++i) {
                std::size_t sz = 1;
                if (i >= leading)
                    sz = input_dims[input_dims.size() - input_rank + (i - leading)];
                const char c = lhs_[i];
                if (char_size.count(c))
                    assertion(char_size[c] == sz, "EinsumDense: inconsistent char size in equation.");
                else
                    char_size[c] = sz;
            }

            // Output-only chars (rare): take the size from full_output_shape_.
            for (std::size_t i = 0; i < rhs_.size(); ++i) {
                const char c = rhs_[i];
                if (char_size.count(c) == 0) {
                    const int dim = full_output_shape_[i];
                    assertion(dim > 0, "EinsumDense: cannot infer size of output-only char.");
                    char_size[c] = static_cast<std::size_t>(dim);
                }
            }
            return char_size;
        }

        static std::vector<std::size_t> sizes_for_chars(const std::string& chars,
            const std::map<char, std::size_t>& char_size)
        {
            std::vector<std::size_t> sizes(chars.size());
            for (std::size_t i = 0; i < chars.size(); ++i)
                sizes[i] = char_size.at(chars[i]);
            return sizes;
        }

        // Computes the contraction value for one fixed output position. The
        // output position is already encoded in `pos` (mutable scratch space
        // shared across calls).
        float_type contract_one(std::map<char, std::size_t>& pos,
            const float_vec& input_vals, const float_vec& kernel_vals,
            const std::vector<std::size_t>& summed_sizes,
            const std::vector<std::size_t>& summed_strides,
            const std::vector<std::size_t>& lhs_strides,
            const std::vector<std::size_t>& kernel_strides) const
        {
            const std::size_t summed_volume = product(summed_sizes);
            float_type acc = 0;
            for (std::size_t s_idx = 0; s_idx < summed_volume; ++s_idx) {
                decode_index(s_idx, summed_, summed_strides, pos);
                const std::size_t lhs_off = encode_offset(lhs_, lhs_strides, pos);
                const std::size_t k_off = encode_offset(rhs_kernel_, kernel_strides, pos);
                acc += input_vals[lhs_off] * kernel_vals[k_off];
            }
            return acc;
        }

        // Adds bias broadcast over the configured bias_axes (a subset of rhs).
        // Keras stores the bias variable with its dims in rhs char order, not
        // in the literal bias_axes string order (e.g. bias_axes='ed' on
        // rhs='abde' still produces a (d, e)-shaped bias). Reorder the chars
        // before computing strides so we read the right element.
        void add_bias(float_vec& out,
            const std::map<char, std::size_t>& char_size,
            const std::vector<std::size_t>& rhs_strides) const
        {
            if (bias_axes_.empty())
                return;
            std::string bias_chars;
            for (char c : rhs_)
                if (bias_axes_.find(c) != std::string::npos)
                    bias_chars.push_back(c);

            const auto& bias_vals = *bias_.as_vector();
            const auto bias_sizes = sizes_for_chars(bias_chars, char_size);
            const auto bias_strides = compute_strides(bias_sizes);

            std::map<char, std::size_t> pos;
            for (std::size_t out_idx = 0; out_idx < out.size(); ++out_idx) {
                decode_index(out_idx, rhs_, rhs_strides, pos);
                out[out_idx] += bias_vals[encode_offset(bias_chars, bias_strides, pos)];
            }
        }

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto char_size = derive_char_sizes(input);

            const auto lhs_sizes = sizes_for_chars(lhs_, char_size);
            const auto kernel_sizes = sizes_for_chars(rhs_kernel_, char_size);
            const auto rhs_sizes = sizes_for_chars(rhs_, char_size);
            const auto summed_sizes = sizes_for_chars(summed_, char_size);

            const auto lhs_strides = compute_strides(lhs_sizes);
            const auto kernel_strides = compute_strides(kernel_sizes);
            const auto rhs_strides = compute_strides(rhs_sizes);
            const auto summed_strides = compute_strides(summed_sizes);

            const std::size_t out_volume = product(rhs_sizes);
            const auto& input_vals = *input.as_vector();
            const auto& kernel_vals = *kernel_.as_vector();

            float_vec out(out_volume, 0);
            std::map<char, std::size_t> pos;
            for (std::size_t out_idx = 0; out_idx < out_volume; ++out_idx) {
                decode_index(out_idx, rhs_, rhs_strides, pos);
                out[out_idx] = contract_one(pos, input_vals, kernel_vals,
                    summed_sizes, summed_strides, lhs_strides, kernel_strides);
            }

            add_bias(out, char_size, rhs_strides);

            // Drop the leading batch char (its size in fdeep is always 1).
            assertion(rhs_sizes.size() >= 2,
                "EinsumDense output equation must have at least a batch char and one feature char.");
            std::vector<std::size_t> out_dims_no_batch(rhs_sizes.begin() + 1, rhs_sizes.end());
            return { tensor(create_tensor_shape_from_dims(out_dims_no_batch), std::move(out)) };
        }
    };

}
}
