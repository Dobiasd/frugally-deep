// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
// Since "batch size" is always 1 it simply scales and shifts the input tensor.
class batch_normalization_layer : public layer
{
public:
    explicit batch_normalization_layer(const std::string& name, float_t epsilon,
        const float_vec& beta, const float_vec& gamma)
        : layer(name),
        epsilon_(epsilon),
        beta_(beta),
        gamma_(gamma)
    {
    }
protected:
    float_t epsilon_;
    float_vec beta_;
    float_vec gamma_;
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of tensors");
        const auto& input = inputs[0];

        if (input.size().depth_ == 1 && input.size().height_ == 1)
        {
            auto result = input;
            if (!gamma_.empty())
            {
                assertion(result.size().width_ == gamma_.size(),
                    "invalid gamma");
                result = matrix3d(result.size(),
                    fplus::zip_with(std::multiplies<float_t>(),
                        result.as_vector(), gamma_));
            }
            if (!beta_.empty())
            {
                assertion(result.size().width_ == beta_.size(), "invalid beta");
                result = matrix3d(result.size(),
                    fplus::zip_with(std::plus<float_t>(),
                        result.as_vector(), beta_));
            }
            return {result};
        }

        auto slices = matrix3d_to_depth_slices(input);
        if (!gamma_.empty())
        {
            assertion(slices.size() == gamma_.size(), "invalid gamma");
            slices = fplus::zip_with(multiply_matrix2d_elems, slices, gamma_);
        }
        if (!beta_.empty())
        {
            assertion(slices.size() == beta_.size(), "invalid beta");
            slices = fplus::zip_with(add_to_matrix2d_elems, slices, beta_);
        }
        return {matrix3d_from_depth_slices(slices)};
    }
};

} // namespace fd
