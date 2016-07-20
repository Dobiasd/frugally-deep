// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class unpool_layer : public layer
{
public:
    explicit unpool_layer(const size3d& size_in, std::size_t scale_factor) :
        size_in_(size_in), scale_factor_(scale_factor)
    {
    }
    std::size_t param_count() const override
    {
        return 0;
    }
    float_vec get_params() const override
    {
        return {};
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
    }
    const size3d& input_size() const override
    {
        return size_in_;
    }
    size3d output_size() const override
    {
        return size3d(
            size_in_.depth_,
            size_in_.height_ * scale_factor_,
            size_in_.width_ * scale_factor_);
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return unpool(input);
    }
    size3d size_in_;
    std::size_t scale_factor_;
    matrix3d unpool(const matrix3d& in_vol) const
    {
        matrix3d out_vol(
            size3d(
                in_vol.size().depth_,
                in_vol.size().height_ * scale_factor_,
                in_vol.size().width_ * scale_factor_));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < out_vol.size().height_; ++y)
            {
                std::size_t y_in = y / scale_factor_;
                for (std::size_t x = 0; x < out_vol.size().width_; ++x)
                {
                    std::size_t x_in = x / scale_factor_;
                    float_t val = in_vol.get(z, y_in, x_in);
                    for (std::size_t yf = 0; yf < scale_factor_; ++yf)
                    {
                        for (std::size_t xf = 0; xf < scale_factor_; ++xf)
                        {
                            out_vol.set(z, y, x, val);
                        }
                    }
                }
            }
        }
        return out_vol;
    }
};

} // namespace fd
