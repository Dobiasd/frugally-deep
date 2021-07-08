// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/pooling_2d_layer.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

FDEEP_FORCE_INLINE tensor average_pool_2d(
    std::size_t pool_height, std::size_t pool_width,
    std::size_t strides_y, std::size_t strides_x,
    bool channels_first,
    padding pad_type,
    const tensor& in)
{
    const float_type invalid = std::numeric_limits<float_type>::lowest();

    const std::size_t feature_count = channels_first
        ? in.shape().height_
        : in.shape().depth_
        ;

    const std::size_t in_height = channels_first
        ? in.shape().width_
        : in.shape().height_
        ;

    const std::size_t in_width = channels_first
        ? in.shape().depth_
        : in.shape().width_
        ;

    const auto conv_cfg = preprocess_convolution(
        shape2(pool_height, pool_width),
        shape2(strides_y, strides_x),
        pad_type, in_height, in_width);

    int pad_top_int = static_cast<int>(conv_cfg.pad_top_);
    int pad_left_int = static_cast<int>(conv_cfg.pad_left_);
    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    if (channels_first)
    {
        tensor out(
            tensor_shape_with_changed_rank(
                tensor_shape(feature_count, out_height, out_width),
                in.shape().rank()),
            0);

        for (std::size_t z = 0; z < feature_count; ++z)
        {
            for (std::size_t y = 0; y < out_height; ++y)
            {
                for (std::size_t x = 0; x < out_width; ++x)
                {
                    float_type val = 0;
                    std::size_t divisor = 0;
                    for (std::size_t yf = 0; yf < pool_height; ++yf)
                    {
                        int in_get_y = static_cast<int>(strides_y * y + yf) - pad_top_int;
                        for (std::size_t xf = 0; xf < pool_width; ++xf)
                        {
                            int in_get_x = static_cast<int>(strides_x * x + xf) - pad_left_int;
                            const auto current = in.get_x_z_padded(invalid, z, in_get_y, in_get_x);
                            if (current != invalid)
                            {
                                val += current;
                                divisor += 1;
                            }
                        }
                    }

                    out.set_ignore_rank(tensor_pos(z, y, x), val / static_cast<float_type>(divisor));
                }
            }
        }
        return out;
    }
    else
    {
        tensor out(
            tensor_shape_with_changed_rank(
                tensor_shape(out_height, out_width, feature_count),
                in.shape().rank()),
            0);

        for (std::size_t y = 0; y < out_height; ++y)
        {
            for (std::size_t x = 0; x < out_width; ++x)
            {
                for (std::size_t z = 0; z < feature_count; ++z)
                {
                    float_type val = 0;
                    std::size_t divisor = 0;
                    for (std::size_t yf = 0; yf < pool_height; ++yf)
                    {
                        int in_get_y = static_cast<int>(strides_y * y + yf) - pad_top_int;
                        for (std::size_t xf = 0; xf < pool_width; ++xf)
                        {
                            int in_get_x = static_cast<int>(strides_x * x + xf) - pad_left_int;
                            const auto current = in.get_y_x_padded(invalid,
                                in_get_y, in_get_x, z);
                            if (current != invalid)
                            {
                                val += current;
                                divisor += 1;
                            }
                        }
                    }

                    out.set_ignore_rank(tensor_pos(y, x, z), val / static_cast<float_type>(divisor));
                }
            }
        }
        return out;
    }
}

class average_pooling_2d_layer : public pooling_2d_layer
{
public:
    explicit average_pooling_2d_layer(const std::string& name,
        const shape2& pool_size, const shape2& strides, bool channels_first,
        padding p) :
        pooling_2d_layer(name, pool_size, strides, channels_first, p)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {
        if (pool_size_ == shape2(2, 2) && strides_ == shape2(2, 2))
            return average_pool_2d(2, 2, 2, 2, channels_first_, padding_, in);
        else if (pool_size_ == shape2(4, 4) && strides_ == shape2(4, 4))
            return average_pool_2d(4, 4, 4, 4, channels_first_, padding_, in);
        else
            return average_pool_2d(
                pool_size_.height_, pool_size_.width_,
                strides_.height_, strides_.width_,
                channels_first_, padding_, in);
    }
};

} } // namespace fdeep, namespace internal
