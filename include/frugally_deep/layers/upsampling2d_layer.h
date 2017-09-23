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

namespace fdeep
{

class upsampling2d_layer : public layer
{
public:
    explicit upsampling2d_layer(const std::string& name, std::size_t scale_factor) :
        layer(name),
        scale_factor_(scale_factor)
    {
        assert(scale_factor_ % 2 == 0);
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override final
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {upsampling2d(input)};
    }
    const std::size_t scale_factor_;
    tensor3 upsampling2d(const tensor3& in_vol) const
    {
        tensor3 out_vol(
            shape3(
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
};

} // namespace fdeep
