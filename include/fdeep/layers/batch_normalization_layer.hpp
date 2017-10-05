// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

namespace fdeep { namespace internal
{

// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
// https://stackoverflow.com/a/46444452/1866775
class batch_normalization_layer : public layer
{
public:
    explicit batch_normalization_layer(const std::string& name,
        const float_vec& moving_mean,
        const float_vec& moving_variance,
        const float_vec& beta,
        const float_vec& gamma,
        float_t epsilon)
        : layer(name),
        moving_mean_(moving_mean),
        moving_variance_(moving_variance),
        beta_(beta),
        gamma_(gamma),
        epsilon_(epsilon)
    {
    }
protected:
    float_vec moving_mean_;
    float_vec moving_variance_;
    float_vec beta_;
    float_vec gamma_;
    float_t epsilon_;

    tensor3 apply_to_slices(const tensor3& input) const
    {
        const auto slices = tensor3_to_depth_slices(input);

        assertion(moving_mean_.size() == slices.size(), "invalid beta");
        assertion(moving_variance_.size() == slices.size(), "invalid beta");

        if (!gamma_.empty())
        {
            assertion(gamma_.size() == slices.size(), "invalid gamma");
        }
        if (!beta_.empty())
        {
            assertion(beta_.size() == slices.size(), "invalid beta");
        }

        std::vector<tensor2> result;
        result.reserve(slices.size());
        for (std::size_t z = 0; z < slices.size(); ++z)
        {
            tensor2 slice = slices[z];
            slice = sub_from_tensor2_elems(slice, moving_mean_[z]);
            if (!gamma_.empty())
                slice = multiply_tensor2_elems(slice, gamma_[z]);
            slice = divide_tensor2_elems(slice,
                std::sqrt(moving_variance_[z] + epsilon_));
            if (!beta_.empty())
                slice = add_to_tensor2_elems(slice, beta_[z]);
            result.push_back(slice);
        }
        return tensor3_from_depth_slices(result);
    }

    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of tensors");
        const auto& input = inputs.front();
        if (input.shape().depth_ == 1 && input.shape().height_ == 1)
        {
            const auto stack =
                reshape_tensor3(input, shape3(input.shape().width_, 1, 1));
            const auto output = apply_to_slices(stack);
            return
                {reshape_tensor3(output, shape3(1, 1, input.shape().width_))};
        }
        return {apply_to_slices(input)};
    }
};

} } // namespace fdeep, namespace internal
