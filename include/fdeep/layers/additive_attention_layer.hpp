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

    class additive_attention_layer : public layer {
    public:
        explicit additive_attention_layer(const std::string& name, const float_vec& scale)
            : layer(name)
            , scale_(scale)
        {
        }

    protected:
        tensors apply_impl(const tensors& input) const override
        {
            assertion(input.size() == 2 || input.size() == 3, "Invalid number of inputs for Attention layer.");
            const tensor& query = input[0];
            const tensor& value = input[1];
            const tensor& key = input.size() > 2 ? input[2] : value;
            const tensor scores = reshape(
                sum_depth(
                    mult_tensors(tensor(tensor_shape(scale_.size()), float_vec(scale_)),
                        transform_tensor(tanh_typed,
                            add_tensors(
                                reshape(query, tensor_shape(query.shape().width_, 1, query.shape().depth_)),
                                reshape(key, tensor_shape(1, key.shape().width_, key.shape().depth_)))))),
                tensor_shape(query.shape().width_, key.shape().width_));
            const tensor distribution = softmax(scores);
            return { dot_product_tensors(distribution, value, std::vector<int>({ 2, 1 }), false) };
        }
        float_vec scale_;
    };

}
}
