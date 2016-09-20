// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class unpool_layer : public layer
{
public:
    explicit unpool_layer(const size3d& size_in, std::size_t scale_factor) :
        layer(size_in, size3d(
            size_in.depth_,
            size_in.height_ * scale_factor,
            size_in.width_ * scale_factor)),
        scale_factor_(scale_factor)
    {
        assert(scale_factor_ % 2 == 0);
    }
    std::size_t param_count() const override
    {
        return 0;
    }
    float_vec get_params() const override
    {
        return {};
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
    }
    void random_init_params() override
    {
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return unpool(input);
    }
    const std::size_t scale_factor_;
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec&) const override
    {
        return pool(input);
    }
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
                    out_vol.set(z, y, x, in_vol.get(z, y_in, x_in));
                }
            }
        }
        return out_vol;
    }
    matrix3d pool(const matrix3d& in_vol) const
    {
        matrix3d out_vol(
            size3d(
                in_vol.size().depth_,
                in_vol.size().height_ / scale_factor_,
                in_vol.size().width_ / scale_factor_));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < out_vol.size().height_; ++y)
            {
                std::size_t y_in = y * scale_factor_;
                for (std::size_t x = 0; x < out_vol.size().width_; ++x)
                {
                    std::size_t x_in = x * scale_factor_;
                    float_t acc = 0;
                    for (std::size_t yf = 0; yf < scale_factor_; ++yf)
                    {
                        for (std::size_t xf = 0; xf < scale_factor_; ++xf)
                        {
                            acc += in_vol.get(z, y_in + yf, x_in + xf);
                        }
                    }
                    out_vol.set(z, y, x, acc);
                }
            }
        }
        return out_vol;
    }
};

} // namespace fd
