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
        int axis,
        const float_vec& moving_mean,
        const float_vec& moving_variance,
        const float_vec& beta,
        const float_vec& gamma,
        float_type epsilon)
        : layer(name),
        axis_(axis),
        moving_mean_(moving_mean),
        moving_variance_(moving_variance),
        beta_(beta),
        gamma_(gamma),
        epsilon_(epsilon)
    {
    }
protected:
    int axis_;
    float_vec moving_mean_;
    float_vec moving_variance_;
    float_vec beta_;
    float_vec gamma_;
    float_type epsilon_;

    static float_type apply_to_value_gamma_beta(
        const float_vec& moving_mean, const float_vec& beta, const float_vec& gamma,
        float_type val, float_type denom, std::size_t z)
    {
        val -= moving_mean[z];
        val *= gamma[z];
        val /= denom;
        val += beta[z];
        return val;
    }

    static float_type apply_to_value_gamma(
        const float_vec& moving_mean, const float_vec&, const float_vec& gamma,
        float_type val, float_type denom, std::size_t z)
    {
        val -= moving_mean[z];
        val *= gamma[z];
        val /= denom;
        return val;
    }

    static float_type apply_to_value_beta(
        const float_vec& moving_mean, const float_vec& beta, const float_vec&,
        float_type val, float_type denom, std::size_t z)
    {
        val -= moving_mean[z];
        val /= denom;
        val += beta[z];
        return val;
    }

    static float_type apply_to_value(
        const float_vec& moving_mean, const float_vec&, const float_vec&,
        float_type val, float_type denom, std::size_t z)
    {
        val -= moving_mean[z];
        val /= denom;
        return val;
    }

    template <typename F>
    static void apply_to_channel(const F f,
        const float_vec& moving_mean, const float_vec& beta, const float_vec& gamma,
        const tensor& input, tensor& output, float_type denom, std::size_t z, std::size_t dim5, std::size_t dim4)
    {
        for (std::size_t y = 0; y < output.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < output.shape().width_; ++x)
            {
                output.set_ignore_rank(tensor_pos(dim5, dim4, y, x, z),
                    f(moving_mean, beta, gamma,
                        input.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)), denom, z));
            }
        }
    }

    tensor apply_to_slices(const tensor& input) const
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

        tensor output(input.shape(), 0);
        for (std::size_t dim5 = 0; dim5 < output.shape().size_dim_5_; ++dim5)
        {
            for (std::size_t dim4 = 0; dim4 < output.shape().size_dim_4_; ++dim4)
            {
                for (std::size_t z = 0; z < output.shape().depth_; ++z)
                {
                    const float_type denom = std::sqrt(moving_variance_[z] + epsilon_);
                    if (use_gamma && use_beta) {
                        apply_to_channel(apply_to_value_gamma_beta, moving_mean_, beta_, gamma_, input, output, denom, z, dim5, dim4);
                    }
                    else if (use_gamma) {
                        apply_to_channel(apply_to_value_gamma, moving_mean_, beta_, gamma_, input, output, denom, z, dim5, dim4);
                    }
                    else if (use_beta) {
                        apply_to_channel(apply_to_value_beta, moving_mean_, beta_, gamma_, input, output, denom, z, dim5, dim4);
                    }
                    else {
                        apply_to_channel(apply_to_value, moving_mean_, beta_, gamma_, input, output, denom, z, dim5, dim4);
                    }
                }
            }
        }
        return output;
    }

    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        const int adjusted_axis =
            axis_ == -1
            ? 5
            : 5 + axis_ - static_cast<int>(input.shape().rank());

        if (adjusted_axis == 5)
        {
            return {apply_to_slices(input)};
        }
        else if (adjusted_axis == 4)
        {
            return {tensor_with_changed_rank(
                permute_tensor(apply_to_slices(permute_tensor(
                    tensor_with_changed_rank(input, 5),
                    {1, 2, 3, 5, 4})),
                    {1, 2, 3, 5, 4}), input.shape().rank())};
        }
        else if (adjusted_axis == 3)
        {
            return {tensor_with_changed_rank(
                permute_tensor(apply_to_slices(permute_tensor(
                    tensor_with_changed_rank(input, 5),
                    {1, 2, 5, 4, 3})),
                    {1, 2, 5, 4, 3}), input.shape().rank())};
        }
        else if (adjusted_axis == 2)
        {
            return {tensor_with_changed_rank(
                permute_tensor(apply_to_slices(permute_tensor(
                    tensor_with_changed_rank(input, 5),
                    {1, 5, 3, 4, 2})),
                    {1, 5, 3, 4, 2}), input.shape().rank())};
        }
        else if (adjusted_axis == 1)
        {
            return {tensor_with_changed_rank(
                permute_tensor(apply_to_slices(permute_tensor(
                    tensor_with_changed_rank(input, 5),
                    {5, 2, 3, 4, 1})),
                    {5, 2, 3, 4, 1}), input.shape().rank())};
        }
        else {
            raise_error("Invalid axis for batch normalization.");
            // Just to make the compiler happy.
            // In reality, this is never called.
            return inputs;
        }

    }
};

} } // namespace fdeep, namespace internal
