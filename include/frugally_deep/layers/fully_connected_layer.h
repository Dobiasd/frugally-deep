// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include "frugally_deep/matrix2d.h"

namespace fd
{

class fully_connected_layer : public layer
{
public:
    fully_connected_layer(std::size_t n_in, std::size_t n_out)
        : size_in_(size3d(1, 1, n_in)),
            size_out_(1, 1, n_out),
            params_(size2d(n_out, n_in))
    {
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        matrix3d output(output_size());
        for (std::size_t x_out = 0; x_out < output.size().width(); ++x_out)
        {
            float_t out_val = 0;
            for (std::size_t x_in = 0; x_in < input.size().width(); ++x_in)
            {
                out_val += params_.get(x_out, x_in) * input.get(0, 0, x_in);
            }
            output.set(0, 0, x_out, out_val);
        }
        return output;
    }
    std::size_t param_count() const override
    {
        return params_.size().height() * params_.size().width();
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
};

} // namespace fd
