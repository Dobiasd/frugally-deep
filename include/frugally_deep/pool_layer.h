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

namespace internal
{
    class pool_layer : public layer
    {
    public:
        explicit pool_layer(std::size_t scale_factor) :
            scale_factor_(scale_factor)
        {
        }
        matrix3d forward_pass(const matrix3d& input) const override
        {
            return pool(input);
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
        virtual matrix3d pool(const matrix3d& input) const = 0;

        template <typename AccPixelFunc, typename FinalizePixelFunc>
        static matrix3d pool_helper(
                std::size_t scale_factor,
                float_t acc_init,
                AccPixelFunc acc_pixel_func,
                FinalizePixelFunc finalize_pixel_func,
                const matrix3d& in_vol)
        {
            // todo: code mit max_pool_layer teilen
            assert(in_vol.size().height() % scale_factor == 0);
            assert(in_vol.size().width() % scale_factor == 0);
            matrix3d out_vol(
                size3d(
                    in_vol.size().depth(),
                    in_vol.size().height() / scale_factor,
                    in_vol.size().width() / scale_factor));
            for (std::size_t z = 0; z < in_vol.size().depth(); ++z)
            {
                for (std::size_t y = 0; y < out_vol.size().height(); ++y)
                {
                    std::size_t y_in = y * scale_factor;
                    for (std::size_t x = 0; x < out_vol.size().width(); ++x)
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
} // namespace internal

class avg_pool_layer : public internal::pool_layer
{
public:
    explicit avg_pool_layer(std::size_t scale_factor) :
            pool_layer(scale_factor)
    {
    }
private:
    matrix3d pool(const matrix3d& in_vol) const override
    {
        float_t pool_helper_acc_init = 0;
        auto pool_helper_add = [](float_t& acc, float_t val) -> void
        {
            acc += val;
        };
        auto pool_helper_div = [this](float_t acc) -> float_t
        {
            return acc / scale_factor_ * scale_factor_;
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            pool_helper_add,
            pool_helper_div,
            in_vol);
    }
};

class max_pool_layer : public internal::pool_layer
{
public:
    explicit max_pool_layer(std::size_t scale_factor) :
            pool_layer(scale_factor)
    {
    }
private:
    float_t pool_helper_max(float_t& acc, float_t v)
    {
        return std::max(acc, v);
    }
    matrix3d pool(const matrix3d& in_vol) const override
    {
        float_t pool_helper_acc_init = std::numeric_limits<float>::min();
        auto pool_helper_max = [](float_t& acc, float_t val) -> void
        {
            acc = std::max(acc, val);
        };
        auto pool_helper_identity = [this](float_t acc) -> float_t
        {
            return acc / scale_factor_ * scale_factor_;
        };
        return pool_helper(
            scale_factor_,
            pool_helper_acc_init,
            pool_helper_max,
            pool_helper_identity,
            in_vol);
    }
};

} // namespace fd
