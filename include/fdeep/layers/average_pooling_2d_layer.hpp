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

FDEEP_FORCE_INLINE tensor3 average_pool_2d(
    std::size_t pool_height, std::size_t pool_width,
    std::size_t strides_y, std::size_t strides_x,
    padding pad_type,
    bool use_offset,
    const tensor3& in)
{
    const float_type invalid = std::numeric_limits<float_type>::lowest();

    const auto conv_cfg = preprocess_convolution(
        shape_hw(pool_height, pool_width),
        shape_hw(strides_y, strides_x),
        pad_type, use_offset, in.shape());

    int pad_top_int = static_cast<int>(conv_cfg.pad_top_);
    int pad_left_int = static_cast<int>(conv_cfg.pad_left_);
    const std::size_t offset_y = conv_cfg.offset_y_;
    const std::size_t offset_x = conv_cfg.offset_x_;
    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    tensor3 out(shape_hwc(out_height, out_width, in.shape().depth_), 0);
    for (std::size_t z = 0; z < out.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < out.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out.shape().width_; ++x)
            {
                float_type val = 0;
                std::size_t divisor = 0;
                for (std::size_t yf = 0; yf < pool_height; ++yf)
                {
                    int in_get_y = static_cast<int>(offset_y + strides_y * y + yf) - pad_top_int;
                    for (std::size_t xf = 0; xf < pool_width; ++xf)
                    {
                        int in_get_x = static_cast<int>(offset_x + strides_x * x + xf) - pad_left_int;
                        const auto current = in.get_x_y_padded(invalid,
                            in_get_y, in_get_x, z);
                        if (current != invalid)
                        {
                            val += current;
                            divisor += 1;
                        }
                    }
                }
                out.set_yxz(y, x, z, val / static_cast<float_type>(divisor));
            }
        }
    }
    return out;
}

class average_pooling_2d_layer : public pooling_2d_layer
{
public:
    explicit average_pooling_2d_layer(const std::string& name,
        const shape_hw& pool_size, const shape_hw& strides, padding p,
        bool padding_valid_uses_offset, bool padding_same_uses_offset) :
        pooling_2d_layer(name, pool_size, strides, p,
            padding_valid_uses_offset, padding_same_uses_offset)
    {
    }
protected:
    tensor3 pool(const tensor3& in) const override
    {
        if (pool_size_ == shape_hw(2, 2) && strides_ == shape_hw(2, 2))
            return average_pool_2d(2, 2, 2, 2, padding_, use_offset(), in);
        else if (pool_size_ == shape_hw(4, 4) && strides_ == shape_hw(4, 4))
            return average_pool_2d(4, 4, 4, 4, padding_, use_offset(), in);
        else
            return average_pool_2d(
                pool_size_.height_, pool_size_.width_,
                strides_.height_, strides_.width_,
                padding_, use_offset(), in);
    }
};

} } // namespace fdeep, namespace internal
