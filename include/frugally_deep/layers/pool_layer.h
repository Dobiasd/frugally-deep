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

// Abstract base class for pooling layers
class pool_layer : public layer
{
public:
    explicit pool_layer(const size3d& size_in, std::size_t scale_factor) :
        layer(size_in, size3d(
            size_in.depth_,
            size_in.height_ / scale_factor,
            size_in.width_ / scale_factor)),
        scale_factor_(scale_factor)
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
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return pool(input);
    }

    const std::size_t scale_factor_;
    virtual matrix3d pool(const matrix3d& input) const = 0;

    template <typename AccPixelFunc, typename FinalizePixelFunc>
    static matrix3d pool_helper(
            std::size_t scale_factor,
            float_t acc_init,
            AccPixelFunc acc_pixel_func,
            FinalizePixelFunc finalize_pixel_func,
            const matrix3d& in_vol)
    {
        assert(in_vol.size().height_ % scale_factor == 0);
        assert(in_vol.size().width_ % scale_factor == 0);
        matrix3d out_vol(
            size3d(
                in_vol.size().depth_,
                in_vol.size().height_ / scale_factor,
                in_vol.size().width_ / scale_factor));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < out_vol.size().height_; ++y)
            {
                std::size_t y_in = y * scale_factor;
                for (std::size_t x = 0; x < out_vol.size().width_; ++x)
                {
                    std::size_t x_in = x * scale_factor;
                    float_t acc = acc_init;
                    for (std::size_t yf = 0; yf < scale_factor; ++yf)
                    {
                        for (std::size_t xf = 0; xf < scale_factor; ++xf)
                        {
                            acc_pixel_func(
                                acc, in_vol.get(z, y_in + yf, x_in + xf));
                        }
                    }
                    out_vol.set(z, y, x, finalize_pixel_func(acc));
                }
            }
        }
        return out_vol;
    }
};

} // namespace fd
