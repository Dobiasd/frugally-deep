// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/softmax_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class attention_layer : public layer
{
public:
    explicit attention_layer(const std::string& name, bool use_scale, const std::string& score_mode,
        float_type scale, float_type concat_score_weight)
        : layer(name), use_scale_(use_scale), score_mode_(score_mode),
            scale_(scale), concat_score_weight_(concat_score_weight)
    {
        assertion(score_mode_ == "dot" || score_mode_ == "concat", "Invalid score_mode for Attention layer.");
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2 or input.size() == 3, "Invalid number of inputs for Attention layer.");
        const tensor& query = input[0];
        const tensor& value = input[1];
        const tensor& key = input.size() > 2 ? input[2] : value;
        const tensor scores = score_mode_ == "dot" ?
            transform_tensor(fplus::multiply_with(scale_),
                dot_product_tensors(query, transpose(key), std::vector<std::size_t>({2, 1}), false))
            :
            // todo: remove python code
            //scores = self.concat_score_weight * tf.reduce_sum(tf.tanh(self.scale * (q_reshaped + k_reshaped)), axis=-1)
            transform_tensor(fplus::multiply_with(concat_score_weight_),
                reduce_sum_axis(
                    transform_tensor(tanh,
                        transform_tensor(fplus::multiply_with(scale_),
                            concatenate_tensors_depth(std::vector<tensor>{query, key})))));
        const tensor distribution = softmax(scores);
        return {dot_product_tensors(distribution, value, std::vector<std::size_t>({2, 1}), false)};
    }
    bool use_scale_;
    std::string score_mode_;
    float_type scale_;
    float_type concat_score_weight_;
};

} } // namespace fdeep, namespace internal
