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

    class bidirectional_layer : public layer {
    public:
        explicit bidirectional_layer(const std::string& name,
            const std::string& merge_mode,
            std::size_t n_units,
            const std::string& activation,
            const std::string& recurrent_activation,
            const std::string& wrapped_layer_type,
            bool use_bias,
            bool reset_after,
            bool return_sequences,
            const float_vec& forward_weights,
            const float_vec& forward_recurrent_weights,
            const float_vec& bias_forward,
            const float_vec& backward_weights,
            const float_vec& backward_recurrent_weights,
            const float_vec& bias_backward)
            : layer(name)
            , merge_mode_(merge_mode)
            , n_units_(n_units)
            , activation_(activation)
            , recurrent_activation_(recurrent_activation)
            , wrapped_layer_type_(wrapped_layer_type)
            , use_bias_(use_bias)
            , reset_after_(reset_after)
            , return_sequences_(return_sequences)
            , forward_weights_(forward_weights)
            , forward_recurrent_weights_(forward_recurrent_weights)
            , bias_forward_(bias_forward)
            , backward_weights_(backward_weights)
            , backward_recurrent_weights_(backward_recurrent_weights)
            , bias_backward_(bias_backward)
        {
        }

    protected:
        tensors apply_impl(const tensors& inputs) const override
        {
            const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor, shape, tensor_shape), inputs);
            assertion(inputs.size() == 1, "Invalid number of input tensors.");
            assertion(inputs.front().shape().rank() == 2,
                "input tensor must have rank 2, but shape is '" + show_tensor_shapes(input_shapes) + "'");

            const tensor& input = inputs.front();
            const tensor input_reversed = reverse_time_series_in_tensor(input);

            tensors result_forward;
            tensors result_backward;

            if (wrapped_layer_type_ == "LSTM") {
                result_forward = lstm_impl(input, n_units_, use_bias_,
                    return_sequences_, false, forward_weights_,
                    forward_recurrent_weights_, bias_forward_,
                    activation_, recurrent_activation_);
                result_backward = lstm_impl(input_reversed, n_units_, use_bias_,
                    return_sequences_, false, backward_weights_,
                    backward_recurrent_weights_, bias_backward_,
                    activation_, recurrent_activation_);
            } else if (wrapped_layer_type_ == "GRU") {
                result_forward = gru_impl(input, n_units_, use_bias_,
                    reset_after_, return_sequences_, false,
                    forward_weights_, forward_recurrent_weights_,
                    bias_forward_, activation_, recurrent_activation_);
                result_backward = gru_impl(input_reversed, n_units_, use_bias_,
                    reset_after_, return_sequences_, false,
                    backward_weights_, backward_recurrent_weights_,
                    bias_backward_, activation_, recurrent_activation_);
            } else if (wrapped_layer_type_ == "SimpleRNN") {
                result_forward = simple_rnn_impl(input, n_units_, use_bias_,
                    return_sequences_, false, forward_weights_,
                    forward_recurrent_weights_, bias_forward_, activation_);
                result_backward = simple_rnn_impl(input_reversed, n_units_, use_bias_,
                    return_sequences_, false, backward_weights_,
                    backward_recurrent_weights_, bias_backward_, activation_);
            } else {
                raise_error("Bidirectional wrapper around layer '" + wrapped_layer_type_ + "' not supported.");
            }

            const tensor result_backward_reversed = return_sequences_
                ? reverse_time_series_in_tensor(result_backward.front())
                : result_backward.front();

            if (merge_mode_ == "concat") {
                return { concatenate_tensors_depth({ result_forward.front(), result_backward_reversed }) };
            } else if (merge_mode_ == "sum") {
                return { sum_tensors({ result_forward.front(), result_backward_reversed }) };
            } else if (merge_mode_ == "mul") {
                return { multiply_tensors({ result_forward.front(), result_backward_reversed }) };
            } else if (merge_mode_ == "ave") {
                return { average_tensors({ result_forward.front(), result_backward_reversed }) };
            }

            raise_error("Bidirectional merge mode '" + merge_mode_ + "' not supported.");
            return {};
        }

        const std::string merge_mode_;
        const std::size_t n_units_;
        const std::string activation_;
        const std::string recurrent_activation_;
        const std::string wrapped_layer_type_;
        const bool use_bias_;
        const bool reset_after_;
        const bool return_sequences_;
        const float_vec forward_weights_;
        const float_vec forward_recurrent_weights_;
        const float_vec bias_forward_;
        const float_vec backward_weights_;
        const float_vec backward_recurrent_weights_;
        const float_vec bias_backward_;
    };

}
}
