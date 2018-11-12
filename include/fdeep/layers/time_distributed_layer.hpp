// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"
#include <nlohmann/json.hpp>

#include <string>
#include <functional>



namespace fdeep
{
namespace internal
{

class time_distributed_layer : public layer
{
public:
    explicit time_distributed_layer(const std::string& name,
                                    const layer_ptr& inner_layer,
                                    const std::size_t td_input_len,
                                    const std::size_t td_output_len)
        : layer(name),
          inner_layer_(inner_layer),
          td_input_len_(td_input_len),
          td_output_len_(td_output_len)
    {
        assertion(td_output_len_ > 1, "Wrong input dimension");
    }
    
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        const tensor5 input = inputs.front();
        tensor5s result_time_step = {};
        std::size_t len_series = 0;
        tensor5s slices = {};
        std::int32_t concat_axis;

        if (td_input_len_ == 2)
        {
            len_series = input.shape().width_;
            slices = tensor5_to_tensor5s_width_slices(input);
        }
        else if(td_input_len_ == 3)
        {
            len_series = input.shape().height_;
            slices = tensor5_to_tensor5s_height_slices(input);
        }
        else if(td_input_len_ == 4)
        {
            len_series = input.shape().size_dim_4_;
            slices = tensor5_to_tensor5s_dim4_slices(input);
        }
        else if(td_input_len_ == 5)
        {
            len_series = input.shape().size_dim_5_;
            slices = tensor5_to_tensor5s_dim5_slices(input);
        }
        else
            raise_error("invalid input dim for TimeDistributed");
        
        if (td_output_len_ == 2)
            concat_axis = 2;
        else if (td_output_len_ == 3)
            concat_axis = 1;
        else if (td_output_len_ == 4)
            concat_axis = 3;
        else if (td_output_len_ == 5)
            concat_axis = 4;
        else
            raise_error("invalid output dim for TimeDistributed");
        
        for (std::size_t i = 0; i < len_series; ++i)
        {
            const auto curr_result = inner_layer_->apply({slices[i]});
            result_time_step.push_back(curr_result.front());
        }

        return {concatenate_tensor5s(result_time_step, concat_axis)};
    }
    
    const layer_ptr inner_layer_;
    const std::size_t td_input_len_;
    const std::size_t td_output_len_;
};

} // namespace internal
} // namespace fdeep
