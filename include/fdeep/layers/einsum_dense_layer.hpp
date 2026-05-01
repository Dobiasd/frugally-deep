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
        std::string equation_;
        std::vector<int> full_output_shape_;
        std::string bias_axes_;
        tensor kernel_;
        tensor bias_;

        std::string lhs_;
        std::string rhs_kernel_;
        std::string rhs_;
        std::string summed_;

        static std::string parse_lhs(const std::string& eq)
        {
            const auto comma = eq.find(',');
            return eq.substr(0, comma);
        }

        static std::string parse_rhs_kernel(const std::string& eq)
        {
            const auto arrow = eq.find("->");
            const auto comma = eq.find(',');
            return eq.substr(comma + 1, arrow - comma - 1);
        }

        static std::string parse_rhs(const std::string& eq)
        {
            const auto arrow = eq.find("->");
            return eq.substr(arrow + 2);
        }

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

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);

            // Determine the size of each char in the equation.
            std::map<char, std::size_t> char_size;

            // Sizes from the kernel: kernel rank == rhs_kernel_.size().
            const auto kernel_dims = kernel_.shape().dimensions();
            const std::size_t kernel_rank = rhs_kernel_.size();
            assertion(kernel_rank <= kernel_dims.size(),
                "EinsumDense: kernel chars exceed kernel dimensions.");
            for (std::size_t i = 0; i < kernel_rank; ++i) {
                const std::size_t sz = kernel_dims[kernel_dims.size() - kernel_rank + i];
                char_size[rhs_kernel_[i]] = sz;
            }

            // Sizes from the input tensor: leading lhs chars beyond the input
            // rank correspond to dropped (None) batch dims and have size 1;
            // the trailing chars correspond to physical dims.
            const auto input_dims_full = input.shape().dimensions();
            const std::size_t input_rank = input.shape().rank();
            const std::size_t lhs_chars = lhs_.size();
            assertion(lhs_chars >= input_rank,
                "EinsumDense: input rank exceeds lhs char count.");
            const std::size_t leading = lhs_chars - input_rank;
            for (std::size_t i = 0; i < lhs_chars; ++i) {
                std::size_t sz = 1;
                if (i >= leading) {
                    const std::size_t phys_idx = input_dims_full.size() - input_rank + (i - leading);
                    sz = input_dims_full[phys_idx];
                }
                const char c = lhs_[i];
                if (char_size.count(c)) {
                    assertion(char_size[c] == sz,
                        "EinsumDense: inconsistent char size in equation.");
                } else {
                    char_size[c] = sz;
                }
            }

            // Sizes from the output shape: for chars present neither in input nor
            // in kernel (rare) we read them from full_output_shape_.
            for (std::size_t i = 0; i < rhs_.size(); ++i) {
                const char c = rhs_[i];
                if (char_size.count(c) == 0) {
                    int dim = full_output_shape_[i];
                    assertion(dim > 0,
                        "EinsumDense: cannot infer size of output-only char.");
                    char_size[c] = static_cast<std::size_t>(dim);
                }
            }

            // Compute per-tensor strides over their chars.
            const auto sizes_for_chars = [&](const std::string& chars) {
                std::vector<std::size_t> sizes(chars.size());
                for (std::size_t i = 0; i < chars.size(); ++i)
                    sizes[i] = char_size[chars[i]];
                return sizes;
            };
            const auto lhs_sizes = sizes_for_chars(lhs_);
            const auto kernel_sizes = sizes_for_chars(rhs_kernel_);
            const auto rhs_sizes = sizes_for_chars(rhs_);
            const auto summed_sizes = sizes_for_chars(summed_);

            const auto lhs_strides = compute_strides(lhs_sizes);
            const auto kernel_strides = compute_strides(kernel_sizes);
            const auto rhs_strides = compute_strides(rhs_sizes);

            std::size_t out_volume = 1;
            for (auto sz : rhs_sizes)
                out_volume *= sz;
            std::size_t summed_volume = 1;
            for (auto sz : summed_sizes)
                summed_volume *= sz;

            // Build per-position char index lookups.
            std::map<char, std::size_t> pos;

            const auto& input_vals = *input.as_vector();
            const auto& kernel_vals = *kernel_.as_vector();

            float_vec out(out_volume, 0);

            // Iterate over all output positions.
            for (std::size_t out_idx = 0; out_idx < out_volume; ++out_idx) {
                // Decode out_idx into char positions for rhs chars.
                {
                    std::size_t rem = out_idx;
                    for (std::size_t i = 0; i < rhs_.size(); ++i) {
                        pos[rhs_[i]] = rem / rhs_strides[i];
                        rem %= rhs_strides[i];
                    }
                }
                float_type acc = 0;
                // Iterate over all summed-axis combinations.
                for (std::size_t s_idx = 0; s_idx < summed_volume; ++s_idx) {
                    {
                        std::size_t rem = s_idx;
                        for (std::size_t i = 0; i < summed_.size(); ++i) {
                            std::size_t stride = 1;
                            for (std::size_t j = i + 1; j < summed_.size(); ++j)
                                stride *= summed_sizes[j];
                            pos[summed_[i]] = rem / stride;
                            rem %= stride;
                        }
                    }
                    std::size_t lhs_offset = 0;
                    for (std::size_t i = 0; i < lhs_.size(); ++i)
                        lhs_offset += pos[lhs_[i]] * lhs_strides[i];
                    std::size_t k_offset = 0;
                    for (std::size_t i = 0; i < rhs_kernel_.size(); ++i)
                        k_offset += pos[rhs_kernel_[i]] * kernel_strides[i];
                    acc += input_vals[lhs_offset] * kernel_vals[k_offset];
                }
                out[out_idx] = acc;
            }

            // Add bias on the configured bias_axes (a subset of rhs chars).
            if (!bias_axes_.empty()) {
                const auto& bias_vals = *bias_.as_vector();
                std::vector<std::size_t> bias_sizes(bias_axes_.size());
                for (std::size_t i = 0; i < bias_axes_.size(); ++i)
                    bias_sizes[i] = char_size[bias_axes_[i]];
                const auto bias_strides = compute_strides(bias_sizes);

                for (std::size_t out_idx = 0; out_idx < out_volume; ++out_idx) {
                    std::size_t rem = out_idx;
                    std::map<char, std::size_t> p;
                    for (std::size_t i = 0; i < rhs_.size(); ++i) {
                        p[rhs_[i]] = rem / rhs_strides[i];
                        rem %= rhs_strides[i];
                    }
                    std::size_t b_offset = 0;
                    for (std::size_t i = 0; i < bias_axes_.size(); ++i)
                        b_offset += p[bias_axes_[i]] * bias_strides[i];
                    out[out_idx] += bias_vals[b_offset];
                }
            }

            // Build the output tensor shape from rhs_sizes minus the leading
            // batch char (its size in fdeep is 1, dropped).
            std::vector<std::size_t> out_dims_no_batch(rhs_sizes.begin() + 1, rhs_sizes.end());
            return { tensor(create_tensor_shape_from_dims(out_dims_no_batch), std::move(out)) };
        }
    };

}
}
