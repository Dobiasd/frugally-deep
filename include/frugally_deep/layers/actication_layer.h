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

// Abstract base class for actication layers
class actication_layer : public layer
{
public:
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return transform_input(input);
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
    virtual matrix3d transform_input(const matrix3d& input) const = 0;

    template <typename ActivationFunc>
    static matrix3d transform_helper(
            ActivationFunc actication_func,
            const matrix3d& in_vol)
    {
        matrix3d out_vol(
            size3d(
                in_vol.size().depth(),
                in_vol.size().height(),
                in_vol.size().width()));
        for (std::size_t z = 0; z < in_vol.size().depth(); ++z)
        {
            for (std::size_t y = 0; y < in_vol.size().height(); ++y)
            {
                for (std::size_t x = 0; x < in_vol.size().width(); ++x)
                {
                    out_vol.set(z, y, x,
                        actication_func(
                            in_vol.get(z, y, x)));
                }
            }
        }
        return out_vol;
    }
};

} // namespace fd
