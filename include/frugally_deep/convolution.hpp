// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/common.hpp"

#include "frugally_deep/filter.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

template <std::size_t strides_y, std::size_t strides_x, std::size_t fy, std::size_t fx>
tensor3 convolve_opt(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t offset_y,
    std::size_t offset_x,
    const std::vector<filter>& filters,
    const tensor3& in)
{
    tensor3 out(shape3(filters.size(), out_height, out_width));

    const std::size_t fz = filters.front().shape().depth_;

    for (std::size_t z = 0; z < out.shape().depth_; ++z)
    {
        const filter& filter = filters[z];
        for (std::size_t y = 0; y < out.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out.shape().width_; ++x)
            {
                float_t sum = 0;
                for (std::size_t zf = 0; zf < fz; ++zf)
                {
                    for (std::size_t yf = 0; yf < fy; ++yf)
                    {
                        for (std::size_t xf = 0; xf < fx; ++xf)
                        {
                            sum += filter.get(zf, yf, xf) *
                                in.get(zf,
                                    offset_y + strides_y * y + yf,
                                    offset_x + strides_x * x + xf);
                        }
                    }
                }
                out.set(z, y, x, sum + filter.get_bias());
            }
        }
    }

    return out;

    // todo: convolve_matrix_mult instead of convolve with loops?
    //     (i.e. use im_to_col and matrix multiplication for performance)
}

inline tensor3 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t offset_y,
    std::size_t offset_x,
    std::size_t fy,
    std::size_t fx,
    const std::vector<filter>& filters,
    const tensor3& in)
{
    tensor3 out(shape3(filters.size(), out_height, out_width));

    const std::size_t fz = filters.front().shape().depth_;

    for (std::size_t z = 0; z < out.shape().depth_; ++z)
    {
        const filter& filter = filters[z];
        for (std::size_t y = 0; y < out.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out.shape().width_; ++x)
            {
                float_t sum = 0;
                for (std::size_t zf = 0; zf < fz; ++zf)
                {
                    for (std::size_t yf = 0; yf < fy; ++yf)
                    {
                        for (std::size_t xf = 0; xf < fx; ++xf)
                        {
                            sum += filter.get(zf, yf, xf) *
                                in.get(zf,
                                    offset_y + strides_y * y + yf,
                                    offset_x + strides_x * x + xf);
                        }
                    }
                }
                out.set(z, y, x, sum + filter.get_bias());
            }
        }
    }

    return out;

    // todo: convolve_matrix_mult instead of convolve with loops?
    //     (i.e. use im_to_col and matrix multiplication for performance)
}

enum class padding { valid, same };

inline tensor3 convolve(
    shape2 strides,
    padding pad_type,
    bool use_offset,
    const std::vector<filter>& filters,
    const tensor3& input)
{
    assertion(filters.size() > 0, "no filters");

    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, shape3), filters),
        "all filters must have the same shape");

    const auto filter_shape = filters.front().shape();

    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int in_height = static_cast<int>(input.shape().height_);
    const int in_width = static_cast<int>(input.shape().width_);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);

    int out_height = fplus::ceil(static_cast<float>(in_height - filter_height + 1) / static_cast<float>(strides_y) - 0.001);
    int out_width = fplus::ceil(static_cast<float>(in_width - filter_width + 1) / static_cast<float>(strides_x) - 0.001);
    int pad_along_height = 0;
    int pad_along_width = 0;

    if (pad_type == padding::same)
    {
        out_height = fplus::ceil(static_cast<float>(in_height) / static_cast<float>(strides_y) - 0.001);
        out_width  = fplus::ceil(static_cast<float>(in_width) / static_cast<float>(strides_x) - 0.001);

        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);
    }
    const int pad_top = pad_along_height / 2;
    const int pad_bottom = pad_along_height - pad_top;
    const int pad_left = pad_along_width / 2;
    const int pad_right = pad_along_width - pad_left;

    int offset_y = 0;
    int offset_x = 0;

    if (use_offset)
    {
        offset_y = ((in_height + pad_top + pad_bottom - filter_height) % strides_y) / 2;
    }
    if (use_offset)
    {
        offset_x = ((in_width + pad_left + pad_right - filter_width) % strides_x) / 2;
    }

    std::size_t out_height_size_t = static_cast<std::size_t>(out_height);
    std::size_t out_width_size_t = static_cast<std::size_t>(out_width);
    std::size_t strides_y_size_t = static_cast<std::size_t>(strides_y);
    std::size_t strides_x_size_t = static_cast<std::size_t>(strides_x);
    std::size_t offset_y_size_t = static_cast<std::size_t>(offset_y);
    std::size_t offset_x_size_t = static_cast<std::size_t>(offset_x);
    std::size_t pad_top_size_t = static_cast<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = static_cast<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = static_cast<std::size_t>(pad_left);
    std::size_t pad_right_size_t = static_cast<std::size_t>(pad_right);

    const auto in_padded = pad_tensor3(
        pad_top_size_t, pad_bottom_size_t, pad_left_size_t, pad_right_size_t,
        input);

    // Allow the compiler to optimize common convolution cases.
    if (strides_y_size_t == 1 && strides_x_size_t == 1 && filter_shape.height_ == 1 && filter_shape.width_ == 1)
        return convolve_opt<1, 1, 1, 1>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);
    if (strides_y_size_t == 1 && strides_x_size_t == 1 && filter_shape.height_ == 3 && filter_shape.width_ == 3)
        return convolve_opt<1, 1, 3, 3>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);
    if (strides_y_size_t == 1 && strides_x_size_t == 1 && filter_shape.height_ == 5 && filter_shape.width_ == 5)
        return convolve_opt<1, 1, 5, 5>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);
    if (strides_y_size_t == 2 && strides_x_size_t == 2 && filter_shape.height_ == 1 && filter_shape.width_ == 1)
        return convolve_opt<2, 2, 1, 1>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);
    if (strides_y_size_t == 2 && strides_x_size_t == 2 && filter_shape.height_ == 3 && filter_shape.width_ == 3)
        return convolve_opt<2, 2, 3, 3>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);
    if (strides_y_size_t == 2 && strides_x_size_t == 2 && filter_shape.height_ == 5 && filter_shape.width_ == 5)
        return convolve_opt<2, 2, 5, 5>(out_height_size_t, out_width_size_t, offset_y_size_t, offset_x_size_t, filters, in_padded);

    return convolve(
        out_height_size_t,
        out_width_size_t,
        strides_y_size_t,
        strides_x_size_t,
        offset_y_size_t,
        offset_x_size_t,
        filter_shape.height_,
        filter_shape.width_,
        filters, in_padded);
}

} } // namespace fdeep, namespace internal