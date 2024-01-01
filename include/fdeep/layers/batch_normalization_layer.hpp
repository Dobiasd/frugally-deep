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

    // https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // https://stackoverflow.com/a/46444452/1866775
    class batch_normalization_layer : public layer {
    public:
        explicit batch_normalization_layer(const std::string& name,
            int axis,
            const float_vec& moving_mean,
            const float_vec& moving_variance,
            const float_vec& beta,
            const float_vec& gamma,
            float_type epsilon)
            : layer(name)
            , axis_(axis)
            , moving_mean_(fplus::make_shared_ref<float_vec>(moving_mean))
            , moving_variance_(fplus::make_shared_ref<float_vec>(moving_variance))
            , beta_(fplus::make_shared_ref<float_vec>(beta))
            , gamma_(fplus::make_shared_ref<float_vec>(gamma))
            , epsilon_(epsilon)
        {
            assertion(moving_variance.size() == moving_mean.size(), "invalid sizes");
            assertion(beta.empty() || beta.size() == moving_mean.size(), "invalid sizes");
            assertion(gamma.empty() || gamma.size() == moving_mean.size(), "invalid sizes");
        }

    protected:
        int axis_;
        shared_float_vec moving_mean_;
        shared_float_vec moving_variance_;
        shared_float_vec beta_;
        shared_float_vec gamma_;
        float_type epsilon_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto input = single_tensor_from_tensors(inputs);

            std::vector<std::size_t> dims(5, 1);
            dims[rank_aligned_axis_to_absolute_axis(input.shape().rank(), axis_) - 1] = moving_mean_->size();
            const tensor_shape params_shape = create_tensor_shape_from_dims(dims);

            return { batch_normalization(
                input,
                broadcast(tensor(params_shape, moving_mean_), input.shape()),
                broadcast(tensor(params_shape, moving_variance_), input.shape()),
                beta_->empty() ? tensor(input.shape(), 0) : broadcast(tensor(params_shape, beta_), input.shape()),
                gamma_->empty() ? tensor(input.shape(), 1) : broadcast(tensor(params_shape, gamma_), input.shape()),
                epsilon_) };
        }
    };

}
}
