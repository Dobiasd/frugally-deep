// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/pooling_3d_layer.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

inline void inner_average_pool(const tensor& in, tensor& out,
    std::size_t pool_size_d4, std::size_t pool_height, std::size_t pool_width,
    std::size_t strides_d4, std::size_t strides_y, std::size_t strides_x,
    std::size_t d4, std::size_t y, std::size_t x, std::size_t z,
    int pad_front_int, int pad_top_int, int pad_left_int)
{
    const float_type invalid = std::numeric_limits<float_type>::lowest();
    float_type val = 0;
    std::size_t divisor = 0;
    for (std::size_t d4f = 0; d4f < pool_size_d4; ++d4f)
    {
        int in_get_d4 = static_cast<int>(strides_d4 * d4 + d4f) - pad_front_int;
        for (std::size_t yf = 0; yf < pool_height; ++yf)
        {
            int in_get_y = static_cast<int>(strides_y * y + yf) - pad_top_int;
            for (std::size_t xf = 0; xf < pool_width; ++xf)
            {
                int in_get_x = static_cast<int>(strides_x * x + xf) - pad_left_int;
                const auto current = in.get_padded(invalid,
                    0, in_get_d4, in_get_y, in_get_x, static_cast<int>(z));
                if (current != invalid)
                {
                    val += current;
                    divisor += 1;
                }
            }
        }
    }
    out.set_ignore_rank(tensor_pos(d4, y, x, z), val / static_cast<float_type>(divisor));
}

class average_pooling_3d_layer : public pooling_3d_layer
{
public:
    explicit average_pooling_3d_layer(const std::string& name,
        const shape3& pool_size, const shape3& strides,
        padding p) :
        pooling_3d_layer(name, pool_size, strides, p,
        &inner_average_pool)
    {
    }
};

} } // namespace fdeep, namespace internal
