// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class unpool_layer : public layer
{
public:
    explicit unpool_layer(std::size_t scale_factor) :
        scale_factor_(scale_factor)
    {
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return unpool(input);
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
    std::size_t input_depth() const override
    {
        // todo: everything goes
        // perhaps instead fplus::maybe<std::size_t> fixed_input_depth()
        return 0;
    }
    std::size_t output_depth() const override
    {
        // todo: same_as_input
        return 0;
    }
protected:
    std::size_t scale_factor_;
    matrix3d unpool(const matrix3d& in_vol) const
    {
        matrix3d out_vol(
            size3d(
                in_vol.size().depth(),
                in_vol.size().height() * scale_factor_,
                in_vol.size().width() * scale_factor_));
        for (std::size_t z = 0; z < in_vol.size().depth(); ++z)
        {
            for (std::size_t y = 0; y < out_vol.size().height(); ++y)
            {
                std::size_t y_in = y / scale_factor_;
                for (std::size_t x = 0; x < out_vol.size().width(); ++x)
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
