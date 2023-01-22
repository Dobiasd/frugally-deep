// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

struct convolution3d_config
{
    std::size_t pad_front_;
    std::size_t pad_back_;
    std::size_t pad_top_;
    std::size_t pad_bottom_;
    std::size_t pad_left_;
    std::size_t pad_right_;
    std::size_t out_size_d4_;
    std::size_t out_height_;
    std::size_t out_width_;
};

inline convolution3d_config preprocess_convolution_3d(
    const shape3& filter_shape,
    const shape3& strides,
    padding pad_type,
    std::size_t input_shape_size_d4,
    std::size_t input_shape_height,
    std::size_t input_shape_width)
{
    const int filter_size_d4 = static_cast<int>(filter_shape.size_dim_4_);
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int in_size_d4 = static_cast<int>(input_shape_size_d4);
    const int in_height = static_cast<int>(input_shape_height);
    const int in_width = static_cast<int>(input_shape_width);
    const int strides_d4 = static_cast<int>(strides.size_dim_4_);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);

    int out_size_d4 = 0;
    int out_height = 0;
    int out_width = 0;

    if (pad_type == padding::same || pad_type == padding::causal)
    {
        out_size_d4 = fplus::ceil(static_cast<float>(in_size_d4) / static_cast<float>(strides_d4) - 0.001);
        out_height = fplus::ceil(static_cast<float>(in_height) / static_cast<float>(strides_y) - 0.001);
        out_width  = fplus::ceil(static_cast<float>(in_width) / static_cast<float>(strides_x) - 0.001);
    }
    else
    {
        out_size_d4 = fplus::ceil(static_cast<float>(in_size_d4 - filter_size_d4 + 1) / static_cast<float>(strides_d4) - 0.001);
        out_height = fplus::ceil(static_cast<float>(in_height - filter_height + 1) / static_cast<float>(strides_y) - 0.001);
        out_width = fplus::ceil(static_cast<float>(in_width - filter_width + 1) / static_cast<float>(strides_x) - 0.001);
    }

    int pad_front = 0;
    int pad_back = 0;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;

    if (pad_type == padding::same)
    {
        int pad_along_d4 = 0;
        int pad_along_height = 0;
        int pad_along_width = 0;

        if (in_size_d4 % strides_d4 == 0)
            pad_along_d4 = std::max(filter_size_d4 - strides_d4, 0);
        else
            pad_along_d4 = std::max(filter_size_d4 - (in_size_d4 % strides_d4), 0);
        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);

        pad_front = pad_along_d4 / 2;
        pad_back = pad_along_d4 - pad_front;
        pad_top = pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
    }
    else if (pad_type == padding::causal)
    {
        pad_front = filter_size_d4 - 1;
        pad_top = filter_height - 1;
        pad_left = filter_width - 1;
    }

    std::size_t out_size_d4_size_t = fplus::integral_cast_throw<std::size_t>(out_size_d4);
    std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
    std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
    std::size_t pad_front_size_t = fplus::integral_cast_throw<std::size_t>(pad_front);
    std::size_t pad_back_size_t = fplus::integral_cast_throw<std::size_t>(pad_back);
    std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
    std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);

    return {pad_front_size_t, pad_back_size_t,
        pad_top_size_t, pad_bottom_size_t,
        pad_left_size_t, pad_right_size_t,
        out_size_d4_size_t, out_height_size_t, out_width_size_t};
}

} } // namespace fdeep, namespace internal
