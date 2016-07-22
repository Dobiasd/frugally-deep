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
        params_(size2d(n_out, n_in)),
        biases_(n_out, 0)
    {
    }
    std::size_t param_count() const override
    {
        return params_.size().area() + biases_.size();
    }
    float_vec get_params() const override
    {
        return fplus::append(params_.as_vector(), biases_);
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
        auto splitted = fplus::split_at_idx(params_.size().area(), params);
        params_ = matrix2d(params_.size(), splitted.first);
        assert(splitted.second.size() == biases_.size());
        biases_ = float_vec(splitted.second);
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        auto output = matrix2d_to_matrix3d(
            multiply(params_, depth_slice(0, input)));
        for (std::size_t x_out = 0; x_out < output.size().height_; ++x_out)
        {
            output.set(0, x_out, 0, output.get(0, x_out, 0) + biases_[x_out]);
        }
        return output;
    }
    matrix2d params_;

    // todo
    // To cover the biases also with the matrix multiplication
    // they should be an additional column in params_
    // instead of a separate member.
    // Input would then need to be padded with an additional trailing one.
    // This will make the backward pass easier.e.
    float_vec biases_;
};

} // namespace fd
