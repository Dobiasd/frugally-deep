// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/softmax_layer.hpp"

#include <string>

namespace fdeep {
namespace internal {

    class attention_layer : public layer {
    public:
        explicit attention_layer(const std::string& name, const std::string& score_mode,
            float_type scale, float_type concat_score_weight)
            : layer(name)
            , score_mode_(score_mode)
            , scale_(scale)
            , concat_score_weight_(concat_score_weight)
        {
            assertion(score_mode_ == "dot" || score_mode_ == "concat", "Invalid score_mode for Attention layer.");
        }

    protected:
        tensors apply_impl(const tensors& input) const override
        {
            assertion(input.size() == 2 || input.size() == 3, "Invalid number of inputs for Attention layer.");
            const tensor& query = input[0];
            const tensor& value = input[1];
            const tensor& key = input.size() > 2 ? input[2] : value;
            const tensor scores = score_mode_ == "dot" ? transform_tensor(fplus::multiply_with(scale_),
                                      dot_product_tensors(query, transpose(key), std::vector<int>({ 2, 1 }), false))
                                                       :
                                                       // https://github.com/keras-team/keras/blob/v2.13.1/keras/layers/attention/attention.py
                transform_tensor(fplus::multiply_with(concat_score_weight_),
                    reshape(
                        sum_depth(
                            transform_tensor(tanh_typed,
                                transform_tensor(fplus::multiply_with(scale_),
                                    add_tensors(
                                        reshape(query, tensor_shape(query.shape().width_, 1, query.shape().depth_)),
                                        reshape(key, tensor_shape(1, key.shape().width_, key.shape().depth_)))))),
                        tensor_shape(query.shape().width_, key.shape().width_)));
            const tensor distribution = softmax(scores);
            return { dot_product_tensors(distribution, value, std::vector<int>({ 2, 1 }), false) };
        }
        std::string score_mode_;
        float_type scale_;
        float_type concat_score_weight_;
    };

}
}
