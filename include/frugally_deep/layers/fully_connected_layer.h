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

class fully_connected_layer : public layer
{
public:
    fully_connected_layer(std::size_t n_in, std::size_t n_out)
        : size_in_(size3d(1, 1, n_in)),
            size_out_(1, 1, n_out),
            params_(size2d(n_out, n_in)),
            biases_(n_out, 0)
    {
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        matrix3d output(output_size());
        for (std::size_t x_out = 0; x_out < output.size().width_; ++x_out)
        {
            float_t out_val = 0;
            for (std::size_t x_in = 0; x_in < input.size().width_; ++x_in)
            {
                out_val += params_.get(x_out, x_in) * input.get(0, 0, x_in);
            }
            out_val += biases_[x_out];
            output.set(0, 0, x_out, out_val);
        }
        return output;
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
    const size3d& input_size() const override
    {
        return size_in_;
    }
    size3d output_size() const override
    {
        return size_out_;
    }
private:
    size3d size_in_;
    size3d size_out_;
    matrix2d params_;
    float_vec biases_;
};

} // namespace fd
