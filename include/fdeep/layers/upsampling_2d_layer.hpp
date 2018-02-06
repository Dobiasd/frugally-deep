// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

class upsampling_2d_layer : public layer
{
public:
    explicit upsampling_2d_layer(const std::string& name,
        const shape2& scale_factor) :
    layer(name),
    scale_factor_(scale_factor)
    {
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of inputs tensors");
        const auto& input = inputs.front();
        return {upsampling2d(input)};
    }
    shape2 scale_factor_;
    tensor3 upsampling2d(const tensor3& in_vol) const
    {
        tensor3 out_vol(shape3(
            in_vol.shape().depth_,
            in_vol.shape().height_ * scale_factor_.height_,
            in_vol.shape().width_ * scale_factor_.width_), 0);
        for (std::size_t z = 0; z < in_vol.shape().depth_; ++z)
        {
            for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
            {
                std::size_t y_in = y / scale_factor_.height_;
                for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
                {
                    std::size_t x_in = x / scale_factor_.width_;
                    out_vol.set(z, y, x, in_vol.get(z, y_in, x_in));
                }
            }
        }
        return out_vol;
    }
};

} } // namespace fdeep, namespace internal
