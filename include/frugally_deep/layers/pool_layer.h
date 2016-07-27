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
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override final
    {
        return pool(input);
    }
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec& params_deltas_acc) const override final
    {
        return pool_backwards(input, params_deltas_acc);
    }

    const std::size_t scale_factor_;
    virtual matrix3d pool(const matrix3d& input) const = 0;
    virtual matrix3d pool_backwards(const matrix3d& input,
        float_vec& params_deltas_acc) const = 0;

    template <typename AccPixelFunc, typename AccPixelFunc2,
            typename FinalizePixelFunc>
    static matrix3d pool_helper(
            std::size_t scale_factor,
            float_t acc_init,
            float_t acc2_init,
            AccPixelFunc acc_pixel_func,
            AccPixelFunc2 acc2_pixel_func,
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
                    float_t acc2 = acc2_init;
                    for (std::size_t yf = 0; yf < scale_factor; ++yf)
                    {
                        for (std::size_t xf = 0; xf < scale_factor; ++xf)
                        {
                            const float_t val =
                                in_vol.get(z, y_in + yf, x_in + xf);
                            acc = acc_pixel_func(acc, val);
                            acc2 = acc2_pixel_func(acc2, val);
                        }
                    }
                    out_vol.set(z, y, x, finalize_pixel_func(acc, acc2));
                }
            }
        }
        return out_vol;
    }

    template <typename FillOutVolSquareFunc>
    matrix3d pool_backwards_helper(
        FillOutVolSquareFunc fill_out_vol_square,
        const matrix3d& err_vol) const
    {
        assert(err_vol.size().height_ * scale_factor_ == last_input_.size().height_);
        assert(err_vol.size().width_ * scale_factor_ == last_input_.size().width_);
        matrix3d out_vol(last_input_.size());
        for (std::size_t z = 0; z < err_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < err_vol.size().height_; ++y)
            {
                std::size_t y_out = y * scale_factor_;
                for (std::size_t x = 0; x < err_vol.size().width_; ++x)
                {
                    std::size_t x_out = x * scale_factor_;
                    const float_t err_val = err_vol.get(z, y, x);
                    fill_out_vol_square(z, y_out, x_out, err_val, out_vol);
                }
            }
        }
        return out_vol;
    }
};

} // namespace fd
