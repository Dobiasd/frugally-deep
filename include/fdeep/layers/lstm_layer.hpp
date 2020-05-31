// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <string>
#include <functional>

namespace fdeep
{
namespace internal
{

class lstm_layer : public layer
{
  public:
    explicit lstm_layer(const std::string& name,
                        std::size_t n_units,
                        const std::string& activation,
                        const std::string& recurrent_activation,
                        const bool use_bias,
                        const bool return_sequences,
                        const bool return_state,
                        const bool stateful,
                        const float_vec& weights,
                        const float_vec& recurrent_weights,
                        const float_vec& bias)
        : layer(name),
          n_units_(n_units),
          activation_(activation),
          recurrent_activation_(recurrent_activation),
          use_bias_(use_bias),
          return_sequences_(return_sequences),
          return_state_(return_state),
          stateful_(stateful),
          weights_(weights),
          recurrent_weights_(recurrent_weights),
          bias_(bias),
          state_h_(stateful ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>()),
          state_c_(stateful ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>()),
          use_avail_input_state_for_stateful_(true)

    {
    }

    void reset_states() override
    {
        if (is_stateful()) {
            state_h_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
            state_c_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
            use_avail_input_state_for_stateful_ = true;
        }
    }

    bool is_stateful() const override
    {
        return stateful_;
    }

  protected:
    tensors apply_impl(const tensors &inputs) const override final
    {
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor, shape, tensor_shape), inputs);
        // ensure that tensor shape is (1, 1, 1, seq_len, n_features)
        assertion(inputs.front().shape().size_dim_5_ == 1
                  && inputs.front().shape().size_dim_4_ == 1
                  && inputs.front().shape().height_ == 1,
                  "size_dim_5, size_dim_4 and height dimension must be 1, but shape is '" + show_tensor_shapes(input_shapes) + "'");
        const auto input = inputs.front();

        assertion(inputs.size() == 1 || inputs.size() == 3,
                "Invalid number of input tensors.");

        // RNN behavior since TF 2.1:
        // If an *initial state input is provided*, this is used always for non-stateful models
        // but only on reset for stateful models (including the very first call)
        // If *no input state is provided*, then initial state is 0 for non-stateful
        // and, for stateful, it carries the state from previous call, unless state-reset, in which case it set to 0
        bool initial_state_provided = inputs.size() == 3;
        bool use_last_state_for_initial_state = is_stateful() && !use_avail_input_state_for_stateful_;
        bool use_input_initial_state = initial_state_provided && !use_last_state_for_initial_state;
        // bool use_zero_initial_state = !use_input_initial_state && !use_last_state_for_initial_state;

        tensor state_h = use_input_initial_state ? inputs[1] :
                        use_last_state_for_initial_state ? state_h_.unsafe_get_just() :
                        tensor(tensor_shape(n_units_), static_cast<float_type>(0)); // use_zero_initial_state

        tensor state_c = use_input_initial_state ? inputs[2] :
                        use_last_state_for_initial_state ? state_c_.unsafe_get_just() :
                        tensor(tensor_shape(n_units_), static_cast<float_type>(0)); // use_zero_initial_state

        const auto result = lstm_impl(input, state_h, state_c,
            n_units_, use_bias_, return_sequences_, return_state_, weights_,
            recurrent_weights_, bias_, activation_, recurrent_activation_);
        if (is_stateful()) {
            state_h_ = state_h;
            state_c_ = state_c;
            use_avail_input_state_for_stateful_ = false;
        }
        return result;
    }

    const std::size_t n_units_;
    const std::string activation_;
    const std::string recurrent_activation_;
    const bool use_bias_;
    const bool return_sequences_;
    const bool return_state_;
    const bool stateful_;
    const float_vec weights_;
    const float_vec recurrent_weights_;
    const float_vec bias_;
    mutable fplus::maybe<tensor> state_h_;
    mutable fplus::maybe<tensor> state_c_;
    mutable bool use_avail_input_state_for_stateful_;
};

} // namespace internal
} // namespace fdeep
