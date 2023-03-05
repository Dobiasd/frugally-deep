// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class category_encoding_layer : public layer
{
public:
    explicit category_encoding_layer(const std::string& name,
        const std::size_t& num_tokens,
        const std::string& output_mode)
        : layer(name),
        num_tokens_(num_tokens),
        output_mode_(output_mode)
    {
        assertion(output_mode_ == "one_hot" || output_mode_ == "multi_hot" || output_mode_ == "count",
            "Unsupported output mode (" + output_mode_ + ").");
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        assertion(inputs.size() == 1, "need exactly one input");
        const auto input = inputs[0];
        assertion(input.shape().rank() == 1, "Tensor of rank 1 required, but shape is '" + show_tensor_shape(input.shape()) + "'");
        
        if (output_mode_ == "one_hot") {
            tensor out(tensor_shape(input.shape().depth_, num_tokens_), float_type(0));
            const auto input_vector = *input.as_vector();
            for (std::size_t y = 0; y < input_vector.size(); ++y) {
                const std::size_t idx = fplus::round<float_type, std::size_t>(input_vector[y]);
                assertion(idx <= num_tokens_, "Invalid input value (> num_tokens).");
                out.set_ignore_rank(tensor_pos(y, idx), 1);
            }
            return {out};
        } else {
            tensor out(tensor_shape(num_tokens_), float_type(0));
            for (const auto& x : *(input.as_vector())) {
                const std::size_t idx = fplus::round<float_type, std::size_t>(x);
                assertion(idx <= num_tokens_, "Invalid input value (> num_tokens).");
                if (output_mode_ == "multi_hot") {
                    out.set_ignore_rank(tensor_pos(idx), 1);
                } else if (output_mode_ == "count") {
                    out.set_ignore_rank(tensor_pos(idx), out.get_ignore_rank(tensor_pos(idx)) + 1);
                }
            }
            return {out};
        }
        
    }
    std::size_t num_tokens_;
    std::string output_mode_;
};

} } // namespace fdeep, namespace internal
