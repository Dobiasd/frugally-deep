// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

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
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        // todo: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        auto slices = matrix3d_to_depth_slices(input);
        slices = fplus::zip_with(multiply_matrix2d_elems, slices, gamma_); // todo + epsilon
        slices = fplus::zip_with(add_to_matrix2d_elems, slices, beta_);
        return {matrix3d_from_depth_slices(slices)};
    }
};

} // namespace fd
