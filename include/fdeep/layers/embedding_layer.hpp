// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <functional>

namespace fdeep
{
namespace internal
{

class embedding_layer : public layer
{
  public:
    explicit embedding_layer(const std::string& name,
                             std::size_t input_dim,
                             std::size_t output_dim,
                             const float_vec& weights)
        : layer(name)
        , input_dim_(input_dim)
        , output_dim_(output_dim)
        , weights_(weights)
    {}

  protected:
    tensors apply_impl(const tensors &inputs) const override final
    {
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor, shape, tensor_shape), inputs);

        // ensure that tensor shape is (1, 1, 1, 1, seq_len)
        assertion(inputs.front().shape().size_dim_5_ == 1
                  && inputs.front().shape().size_dim_4_ == 1
                  && inputs.front().shape().height_ == 1
                  && inputs.front().shape().width_ == 1,
                  "size_dim_5, size_dim_4, height and width dimension must be 1, but shape is '" + show_tensor_shapes(input_shapes) + "'");

        tensors results;
        for (auto&& input : inputs)
        {
            const std::size_t sequence_len = input.shape().depth_;
            float_vec output_vec(sequence_len * output_dim_);
            auto&& it = output_vec.begin();

            for (std::size_t i = 0; i < sequence_len; ++i)
            {
                std::size_t index = static_cast<std::size_t>(input.get(tensor_pos(i)));
                assertion(index < input_dim_, "vocabulary item indices must all be strictly less than the value of input_dim");
                it = std::copy_n(weights_.cbegin() + static_cast<float_vec::const_iterator::difference_type>(index * output_dim_), output_dim_, it);
            }

            results.push_back(tensor(tensor_shape(sequence_len, output_dim_), std::move(output_vec)));
        }
        return results;
    }

    const std::size_t input_dim_;
    const std::size_t output_dim_;
    const float_vec weights_;
};

} // namespace internal
} // namespace fdeep
