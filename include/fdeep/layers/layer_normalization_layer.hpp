// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep {
namespace internal {

    class layer_normalization_layer : public layer {
    public:
        explicit layer_normalization_layer(const std::string& name,
            std::vector<int> axes,
            const float_vec& beta,
            const float_vec& gamma,
            float_type epsilon)
            : layer(name)
            , axes_(axes)
            , beta_(fplus::make_shared_ref<float_vec>(beta))
            , gamma_(fplus::make_shared_ref<float_vec>(gamma))
            , epsilon_(epsilon)
        {
        }

    protected:
        std::vector<int> axes_;
        shared_float_vec beta_;
        shared_float_vec gamma_;
        float_type epsilon_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);

            // https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/normalization/layer_normalization.py#L291-L304
            const auto& input_moments = moments(input, axes_);
            const auto& mean = input_moments.first;
            const auto& variance = input_moments.second;

            std::vector<std::size_t> dims(5, 1);
            tensor_shape input_shape = input.shape();
            input_shape.maximize_rank();
            const auto input_shape_dimensions = input_shape.dimensions();
            for (const auto axis : axes_) {
                const std::size_t pos = rank_aligned_axis_to_absolute_axis(input.shape().rank(), axis) - 1;
                dims[pos] = input_shape_dimensions[pos];
            }
            const tensor_shape params_shape = create_tensor_shape_from_dims(dims);

            return { batch_normalization(
                input,
                mean,
                variance,
                beta_->empty() ? tensor(input.shape(), 0) : broadcast(tensor(params_shape, beta_), input.shape()),
                gamma_->empty() ? tensor(input.shape(), 1) : broadcast(tensor(params_shape, gamma_), input.shape()),
                epsilon_) };
        }
    };

}
}
