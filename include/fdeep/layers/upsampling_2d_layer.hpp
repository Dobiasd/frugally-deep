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
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class upsampling_2d_layer : public layer
{
public:
    explicit upsampling_2d_layer(const std::string& name,
        const shape2& scale_factor, const std::string& interpolation) :
    layer(name),
    scale_factor_(scale_factor),
    interpolation_(interpolation)
    {
        assertion(interpolation == "nearest" || interpolation == "bilinear",
            "Invalid interpolation method: " + interpolation
        );
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of inputs tensors");
        const auto& input = inputs.front();
        if (interpolation_ == "nearest")
        {
            return {upsampling2d_nearest(input)};
        }
        else if (interpolation_ == "bilinear")
        {
            return {upsampling2d_bilinear(input)};
        }
        else
        {
            raise_error("Invalid interpolation method: " + interpolation_);
            return inputs;
        }
    }
    shape2 scale_factor_;
    std::string interpolation_;
    tensor5 upsampling2d_nearest(const tensor5& in_vol) const
    {
        tensor5 out_vol(shape5(1, 1,
            in_vol.shape().height_ * scale_factor_.height_,
            in_vol.shape().width_ * scale_factor_.width_,
            in_vol.shape().depth_), 0);
        for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
        {
            std::size_t y_in = y / scale_factor_.height_;
            for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < in_vol.shape().depth_; ++z)
                {
                    std::size_t x_in = x / scale_factor_.width_;
                    out_vol.set(0, 0, y, x, z, in_vol.get(0, 0, y_in, x_in, z));
                }
            }
        }
        return out_vol;
    }
    float_type get_interpolated_bilinearly(const tensor5& t,
        std::size_t pos_dim_5, std::size_t pos_dim_4,
        float_type y, float_type x, std::size_t z) const
    {
        y = fplus::max(0, y);
        x = fplus::max(0, x);
        y = fplus::min(y, t.height());
        x = fplus::min(x, t.width());
        std::size_t y_top = static_cast<std::size_t>(fplus::max(0, fplus::floor(y)));
        std::size_t y_bottom = static_cast<std::size_t>(fplus::min(t.height() - 1, y_top + 1));
        std::size_t x_left = static_cast<std::size_t>(fplus::max(0, fplus::floor(x)));
        std::size_t x_right = static_cast<std::size_t>(fplus::min(t.width() - 1, x_left + 1));
        const auto val_top_left = t.get(pos_dim_5, pos_dim_4, y_top, x_left, z);
        const auto val_top_right = t.get(pos_dim_5, pos_dim_4, y_top, x_right, z);
        const auto val_bottom_left = t.get(pos_dim_5, pos_dim_4, y_bottom, x_left, z);
        const auto val_bottom_right = t.get(pos_dim_5, pos_dim_4, y_bottom, x_right, z);
        const auto y_factor_top = static_cast<float_type>(y_bottom) - y;
        const auto y_factor_bottom = 1.0 - y_factor_top;
        const auto x_factor_left = static_cast<float_type>(x_right) - x;
        const auto x_factor_right = 1.0 - x_factor_left;
        return static_cast<float_type>(
            y_factor_top * x_factor_left * val_top_left +
            y_factor_top * x_factor_right * val_top_right +
            y_factor_bottom * x_factor_left * val_bottom_left +
            y_factor_bottom * x_factor_right * val_bottom_right);
    }
    tensor5 upsampling2d_bilinear(const tensor5& in_vol) const
    {
        tensor5 out_vol(shape5(1, 1,
            in_vol.shape().height_ * scale_factor_.height_,
            in_vol.shape().width_ * scale_factor_.width_,
            in_vol.shape().depth_), 0);
        for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
        {
            const auto y_in = (static_cast<float_type>(y) + 0.5f) / static_cast<float_type>(scale_factor_.height_) - 0.5f;
            for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < in_vol.shape().depth_; ++z)
                {
                    const auto x_in = (static_cast<float_type>(x) + 0.5f) / static_cast<float_type>(scale_factor_.width_) - 0.5f;
                    out_vol.set(0, 0, y, x, z,
                        get_interpolated_bilinearly(in_vol, 0, 0, y_in, x_in, z));
                }
            }
        }
        return out_vol;
    }
};

} } // namespace fdeep, namespace internal
