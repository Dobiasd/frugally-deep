// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include "frugally_deep/matrix2d.h"

#include "fplus/fplus.h"

namespace fd
{

// Takes a single column volume (size3d(1, n, 1)) as input.
class fully_connected_layer : public layer
{
public:
    fully_connected_layer(std::size_t n_in, std::size_t n_out) :
        layer(size3d(1, n_in, 1), size3d(1, n_out, 1)),
        params_(size2d(n_out, n_in + 1))
    {
    }
    std::size_t param_count() const override
    {
        return params_.size().area();
    }
    float_vec get_params() const override
    {
        return params_.as_vector();
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
        params_ = matrix2d(params_.size(), params);
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        matrix2d input_slice_with_bias_neuron(
            size2d(input.size().height_ + 1, 1),
            fplus::append(input.as_vector(), {1}));
        return matrix2d_to_matrix3d(
            multiply(params_, input_slice_with_bias_neuron));
    }

    matrix3d backward_pass_impl(const matrix3d& input,
        float_deq& params_deltas) const override
    {
        auto output = matrix2d_to_matrix3d(
            multiply_transposed(params_, depth_slice(0, input)));
        return matrix3d(input_size(), fplus::init(output.as_vector()))
    }

    matrix2d params_;
};

} // namespace fd
