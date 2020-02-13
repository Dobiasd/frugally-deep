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

class bidirectional_layer : public layer
{
public:
    explicit bidirectional_layer(const std::string& name,
                        const std::string& merge_mode,
                        const std::size_t n_units,
                        const std::string& activation,
                        const std::string& recurrent_activation,
                        const std::string& wrapped_layer_type,
                        const bool use_bias,
                        const bool reset_after,
                        const bool return_sequences,
                        const bool stateful,
                        const float_vec& forward_weights,
                        const float_vec& forward_recurrent_weights,
                        const float_vec& bias_forward,
                        const float_vec& backward_weights,
                        const float_vec& backward_recurrent_weights,
                        const float_vec& bias_backward
                        )
        : layer(name),
        merge_mode_(merge_mode),
        n_units_(n_units),
        activation_(activation),
        recurrent_activation_(recurrent_activation),
        wrapped_layer_type_(wrapped_layer_type),
        use_bias_(use_bias),
        reset_after_(reset_after),
        return_sequences_(return_sequences),
        stateful_(stateful),
        forward_weights_(forward_weights),
        forward_recurrent_weights_(forward_recurrent_weights),
        bias_forward_(bias_forward),
        backward_weights_(backward_weights),
        backward_recurrent_weights_(backward_recurrent_weights),
        bias_backward_(bias_backward),
        forward_state_h_(stateful ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>()),
        forward_state_c_(stateful && wrapped_layer_type_has_state_c(wrapped_layer_type) ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>()),
        backward_state_h_(stateful ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>()),
        backward_state_c_(stateful && wrapped_layer_type_has_state_c(wrapped_layer_type) ? tensor(tensor_shape(n_units), static_cast<float_type>(0)) : fplus::nothing<tensor>())
        {
    }

    void reset_states() override
     {
        if (is_stateful()) {
            forward_state_h_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
            forward_state_c_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
            backward_state_h_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
            backward_state_c_ = tensor(tensor_shape(n_units_), static_cast<float_type>(0));
        }
     }

     bool is_stateful() const override
     {
         return stateful_;
     }


protected:

    static bool wrapped_layer_type_has_state_c(const std::string& wrapped_layer_type) {
        if (wrapped_layer_type == "LSTM" || wrapped_layer_type == "CuDNNLSTM") {
            return true;
        }
        if (wrapped_layer_type == "GRU" || wrapped_layer_type == "CuDNNGRU") {
            return false;
        }
        raise_error("layer '" + wrapped_layer_type + "' not yet implemented");
        return false;
    }

    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor, shape, tensor_shape), inputs);

        // ensure that tensor shape is (1, 1, 1, seq_len, n_features)
        assertion(inputs.front().shape().rank() == 2,
                  "size_dim_5, size_dim_4 and height dimension must be 1, but shape is '" + show_tensor_shapes(input_shapes) + "'");

        const auto input = inputs.front();

        tensors result_forward = {};
        tensors result_backward = {};
        tensors bidirectional_result = {};

        const tensor input_reversed = reverse_time_series_in_tensor(input);

        if (wrapped_layer_type_ == "LSTM" || wrapped_layer_type_ == "CuDNNLSTM")
        {
            assertion(inputs.size() == 1 || inputs.size() == 5,
                "Invalid number of input tensors.");

            tensor forward_state_h = inputs.size() == 5
            ? inputs[1]
            : is_stateful()
                ? forward_state_h_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            tensor forward_state_c = inputs.size() == 5
            ? inputs[2]
            : is_stateful()
                ? forward_state_c_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            tensor backward_state_h = inputs.size() == 5
            ? inputs[3]
            : is_stateful()
                ? backward_state_h_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            tensor backward_state_c = inputs.size() == 5
            ? inputs[4]
            : is_stateful()
                ? backward_state_c_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            result_forward = lstm_impl(input, forward_state_h, forward_state_c,
                                       n_units_, use_bias_, return_sequences_, stateful_,
                                       forward_weights_, forward_recurrent_weights_,
                                       bias_forward_, activation_, recurrent_activation_);
            result_backward = lstm_impl(input_reversed, backward_state_h, backward_state_c,
                                        n_units_, use_bias_, return_sequences_, stateful_,
                                        backward_weights_, backward_recurrent_weights_,
                                        bias_backward_, activation_, recurrent_activation_);
            if (is_stateful()) {
                forward_state_h_ = forward_state_h;
                forward_state_c_ = forward_state_c;
                backward_state_h_ = backward_state_h;
                backward_state_c_ = backward_state_c;
             }
        }
        else if (wrapped_layer_type_ == "GRU" || wrapped_layer_type_ == "CuDNNGRU")
        {
            assertion(inputs.size() == 1 || inputs.size() == 3,
                "Invalid number of input tensors.");

            tensor forward_state_h = inputs.size() == 3
            ? inputs[1]
            : is_stateful()
                ? forward_state_h_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            tensor backward_state_h = inputs.size() == 3
            ? inputs[2]
            : is_stateful()
                ? backward_state_h_.unsafe_get_just()
                : tensor(tensor_shape(n_units_), static_cast<float_type>(0));

            result_forward = gru_impl(input, forward_state_h, n_units_, use_bias_, reset_after_, return_sequences_, false,
                                      forward_weights_, forward_recurrent_weights_,
                                      bias_forward_, activation_, recurrent_activation_);
            result_backward = gru_impl(input_reversed, backward_state_h, n_units_, use_bias_, reset_after_, return_sequences_, false,
                                       backward_weights_, backward_recurrent_weights_,
                                       bias_backward_, activation_, recurrent_activation_);
            if (is_stateful()) {
                forward_state_h_ = forward_state_h;
                backward_state_h_ = backward_state_h;
             }
        }
        else
            raise_error("layer '" + wrapped_layer_type_ + "' not yet implemented");

        const tensor result_backward_reversed = reverse_time_series_in_tensor(result_backward.front());

        if (merge_mode_ == "concat")
        {
            bidirectional_result = {concatenate_tensors_depth({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "sum")
        {
            bidirectional_result = {sum_tensors({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "mul")
        {
            bidirectional_result = {multiply_tensors({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "ave")
        {
            bidirectional_result = {average_tensors({result_forward.front(), result_backward_reversed})};
        }
        else
            raise_error("merge mode '" + merge_mode_ + "' not valid");

        return bidirectional_result;
    }

    const std::string merge_mode_;
    const std::size_t n_units_;
    const std::string activation_;
    const std::string recurrent_activation_;
    const std::string wrapped_layer_type_;
    const bool use_bias_;
    const bool reset_after_;
    const bool return_sequences_;
    const bool stateful_;
    const float_vec forward_weights_;
    const float_vec forward_recurrent_weights_;
    const float_vec bias_forward_;
    const float_vec backward_weights_;
    const float_vec backward_recurrent_weights_;
    const float_vec bias_backward_;
    mutable fplus::maybe<tensor> forward_state_h_;
    mutable fplus::maybe<tensor> forward_state_c_;
    mutable fplus::maybe<tensor> backward_state_h_;
    mutable fplus::maybe<tensor> backward_state_c_;
};

} // namespace internal
} // namespace fdeep
