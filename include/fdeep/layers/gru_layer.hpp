// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <string>

namespace fdeep {
namespace internal {

    class gru_layer : public layer {
    public:
        explicit gru_layer(const std::string& name,
            std::size_t n_units,
            const std::string& activation,
            const std::string& recurrent_activation,
            bool use_bias,
            bool reset_after,
            bool return_sequences,
            bool return_state,
            const float_vec& weights,
            const float_vec& recurrent_weights,
            const float_vec& bias)
            : layer(name)
            , n_units_(n_units)
            , activation_(activation)
            , recurrent_activation_(recurrent_activation)
            , use_bias_(use_bias)
            , reset_after_(reset_after)
            , return_sequences_(return_sequences)
            , return_state_(return_state)
            , weights_(weights)
            , recurrent_weights_(recurrent_weights)
            , bias_(bias)
        {
        }

    protected:
        tensors apply_impl(const tensors& inputs) const override
        {
            const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor, shape, tensor_shape), inputs);
            assertion(inputs.front().shape().size_dim_5_ == 1
                    && inputs.front().shape().size_dim_4_ == 1
                    && inputs.front().shape().height_ == 1,
                "size_dim_5, size_dim_4 and height dimension must be 1, but shape is '" + show_tensor_shapes(input_shapes) + "'");
            assertion(inputs.size() == 1, "Invalid number of input tensors.");

            return gru_impl(inputs.front(), n_units_, use_bias_, reset_after_,
                return_sequences_, return_state_, weights_,
                recurrent_weights_, bias_, activation_, recurrent_activation_);
        }

        const std::size_t n_units_;
        const std::string activation_;
        const std::string recurrent_activation_;
        const bool use_bias_;
        const bool reset_after_;
        const bool return_sequences_;
        const bool return_state_;
        const float_vec weights_;
        const float_vec recurrent_weights_;
        const float_vec bias_;
    };

}
}
