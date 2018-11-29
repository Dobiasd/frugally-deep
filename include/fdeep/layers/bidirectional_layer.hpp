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
        forward_weights_(forward_weights),
        forward_recurrent_weights_(forward_recurrent_weights),
        bias_forward_(bias_forward),
        backward_weights_(backward_weights),
        backward_recurrent_weights_(backward_recurrent_weights),
        bias_backward_(bias_backward)
    {
    }
    
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor5, shape, shape5), inputs);
        
        // ensure that tensor5 shape is (1, 1, 1, seq_len, n_features)
        assertion(inputs.front().shape().size_dim_5_ == 1
                  && inputs.front().shape().size_dim_4_ == 1
                  && inputs.front().shape().height_ == 1,
                  "size_dim_5, size_dim_4 and height dimension must be 1, but shape is '" + show_shape5s(input_shapes) + "'");
        
        const auto input = inputs.front();
        
        tensor5s result_forward = {};
        tensor5s result_backward = {};
        tensor5s bidirectional_result = {};
        
        const tensor5 input_reversed = reverse_time_series_in_tensor5(input);
        
        if (wrapped_layer_type_ == "LSTM")
        {
            result_forward = lstm_impl(input, n_units_, use_bias_, return_sequences_,
                                       forward_weights_, forward_recurrent_weights_,
                                       bias_forward_, activation_, recurrent_activation_);
            result_backward = lstm_impl(input_reversed, n_units_, use_bias_, return_sequences_,
                                        backward_weights_, backward_recurrent_weights_,
                                        bias_backward_, activation_, recurrent_activation_);
        }
        else if (wrapped_layer_type_ == "GRU")
        {
            result_forward = gru_impl(input, n_units_, use_bias_, reset_after_, return_sequences_,
                                      forward_weights_, forward_recurrent_weights_,
                                      bias_forward_, activation_, recurrent_activation_);
            result_backward = gru_impl(input_reversed, n_units_, use_bias_, reset_after_, return_sequences_,
                                       backward_weights_, backward_recurrent_weights_,
                                       bias_backward_, activation_, recurrent_activation_);
        }
        else
            raise_error("layer '" + wrapped_layer_type_ + "' not yet implemented");
            
        const tensor5 result_backward_reversed = reverse_time_series_in_tensor5(result_backward.front());
        
        if (merge_mode_ == "concat")
        {
            bidirectional_result = {concatenate_tensor5s_depth({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "sum")
        {
            bidirectional_result = {sum_tensor5s({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "mul")
        {
            bidirectional_result = {multiply_tensor5s({result_forward.front(), result_backward_reversed})};
        }
        else if (merge_mode_ == "ave")
        {
            bidirectional_result = {average_tensor5s({result_forward.front(), result_backward_reversed})};
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
    const float_vec forward_weights_;
    const float_vec forward_recurrent_weights_;
    const float_vec bias_forward_;
    const float_vec backward_weights_;
    const float_vec backward_recurrent_weights_;
    const float_vec bias_backward_;
};

} // namespace internal
} // namespace fdeep
