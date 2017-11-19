// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/pooling_2d_layer.hpp"

#include <limits>

namespace fdeep { namespace internal
{

class max_pooling_2d_layer : public pooling_2d_layer
{
public:
    explicit max_pooling_2d_layer(const std::string& name,
        const shape2& pool_size, const shape2& strides, padding p,
        bool padding_valid_uses_offset, bool padding_same_uses_offset) :
        pooling_2d_layer(name, pool_size, strides, p,
            padding_valid_uses_offset, padding_same_uses_offset)
    {
    }
protected:
    tensor3 pool(const tensor3& in) const override
    {
        const float_type invalid = std::numeric_limits<float_type>::lowest();
        const auto conv_cfg = preprocess_convolution(
            pool_size_, strides_, padding_, use_offset(), in.shape());

        int pad_top_int = static_cast<int>(conv_cfg.pad_top_);
        int pad_left_int = static_cast<int>(conv_cfg.pad_left_);
        const std::size_t strides_y = strides_.height_;
        const std::size_t strides_x = strides_.width_;
        const std::size_t offset_y = conv_cfg.offset_y_;
        const std::size_t offset_x = conv_cfg.offset_x_;
        const std::size_t out_height = conv_cfg.out_height_;
        const std::size_t out_width = conv_cfg.out_width_;

        tensor3 out(shape3(in.shape().depth_, out_height, out_width), 0);

        for (std::size_t z = 0; z < out.shape().depth_; ++z)
        {
            for (std::size_t y = 0; y < out.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < out.shape().width_; ++x)
                {
                    float_type val = std::numeric_limits<float_type>::lowest();
                    for (std::size_t yf = 0; yf < pool_size_.height_; ++yf)
                    {
                        int in_get_y = static_cast<int>(offset_y + strides_y * y + yf) - pad_top_int;
                        for (std::size_t xf = 0; xf < pool_size_.width_; ++xf)
                        {
                            int in_get_x = static_cast<int>(offset_x + strides_x * x + xf) - pad_left_int;
                            const auto current = in.get_x_y_padded(invalid, z,
                                in_get_y, in_get_x);
                            val = std::max(val, current);
                        }
                    }
                    out.set(z, y, x, val);
                }
            }
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
