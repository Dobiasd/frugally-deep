// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

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
        float_type epsilon)
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
    float_type epsilon_;

    tensor3 apply_to_slices(const tensor3& input) const
    {
        assertion(moving_mean_.size() == input.shape().depth_,
            "invalid beta");
        assertion(moving_variance_.size() == input.shape().depth_,
            "invalid beta");

        const bool use_gamma = !gamma_.empty();
        if (use_gamma)
        {
            assertion(gamma_.size() == input.shape().depth_, "invalid gamma");
        }

        const bool use_beta = !beta_.empty();
        if (use_beta)
        {
            assertion(beta_.size() == input.shape().depth_, "invalid beta");
        }

        tensor3 output(input.shape(), 0);
        for (std::size_t z = 0; z < output.shape().depth_; ++z)
        {
            const float_type denom = std::sqrt(moving_variance_[z] + epsilon_);
            for (std::size_t y = 0; y < output.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < output.shape().width_; ++x)
                {
                    float_type val = input.get_yxz(y, x, z);
                    val -= moving_mean_[z];
                    if (use_gamma)
                        val *= gamma_[z];
                    val /= denom;
                    if (use_beta)
                        val += beta_[z];
                    output.set_yxz(y, x, z, val);
                }
            }
        }
        return output;
    }

    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of tensors");
        const auto& input = inputs.front();
        return {apply_to_slices(input)};
    }
};

} } // namespace fdeep, namespace internal
