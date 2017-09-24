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

inline tensor2 pad_tensor2(
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const tensor2& in)
{
    tensor2 out(shape2(
        in.shape().height_ + padding_top + padding_bottom,
        in.shape().width_ + padding_left + padding_right));
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            out.set(y + padding_top, x + padding_left, in.get(y, x));
        }
    }
    return out;
}

namespace details
{

inline void convolve_go(
    std::size_t strides_y,
    std::size_t strides_x,
    const tensor2& filter,
    const tensor2& in,
    tensor2& out)
{
    const std::size_t fy = filter.shape().height_;
    const std::size_t fx = filter.shape().width_;
    for (std::size_t y = 0; y < out.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < out.shape().width_; ++x)
        {
            for (std::size_t yf = 0; yf < fy; ++yf)
            {
                for (std::size_t xf = 0; xf < fx; ++xf)
                {
                    const float_t add_val = filter.get(yf, xf) *
                        in.get(strides_y * y + yf, strides_x * x + xf);
                    out.set(y, x, out.get(y, x) + add_val);
                }
            }
        }
    }
}

// Give the compiler a chance to unroll the inner loops.
// In tests with a 3x3 filter and clang++ -O3
// the performance was increased by a factor of two by this.
template <
    std::size_t stride_y,
    std::size_t stride_x,
    std::size_t fy,
    std::size_t fx
    >
void convolve_go_template(
    const tensor2& filter,
    const tensor2& in,
    tensor2& out)
{
    assertion(filter.shape().height_ == fy, "invalid filter height");
    assertion(filter.shape().width_ == fx, "invalid filter width");
    for (std::size_t y = 0; y < out.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < out.shape().width_; ++x)
        {
            for (std::size_t yf = 0; yf < fy; ++yf)
            {
                for (std::size_t xf = 0; xf < fx; ++xf)
                {
                    const float_t add_val = filter.get(yf, xf) *
                        in.get(stride_y * y + yf, stride_x * x + xf);
                    out.set(y, x, out.get(y, x) + add_val);
                }
            }
        }
    }
}

} // namespace details

inline tensor2 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const tensor2& filter,
    const tensor2& in_orig)
{
    assertion(strides_x * strides_y > 0, "invalid stride");
    const std::size_t h1 = in_orig.shape().height_;
    const std::size_t w1 = in_orig.shape().width_;
    const std::size_t fy = filter.shape().height_;
    const std::size_t fx = filter.shape().width_;

    assertion(fy <= h1, "filter height too large for data");
    assertion(fx <= w1, "filter width too large for data");

    const tensor2 in_padded = pad_tensor2(
        padding_top, padding_bottom, padding_left, padding_right, in_orig);

    tensor2 out(shape2(out_height, out_width));

    if (strides_y == 1 && strides_x == 1 && fy == 1 && fx == 1)
        details::convolve_go_template<1, 1, 1, 1>(filter, in_padded, out);

    else if (strides_y == 1 && strides_x == 1 && fy == 3 && fx == 3)
        details::convolve_go_template<1, 1, 3, 3>(filter, in_padded, out);
    else if (strides_y == 1 && strides_x == 1 && fy == 5 && fx == 5)
        details::convolve_go_template<1, 1, 5, 5>(filter, in_padded, out);

    else if (strides_y == 2 && strides_x == 2 && fy == 1 && fx == 1)
        details::convolve_go_template<2, 2, 1, 1>(filter, in_padded, out);

    else if (strides_y == 2 && strides_x == 2 && fy == 3 && fx == 3)
        details::convolve_go_template<2, 2, 3, 3>(filter, in_padded, out);
    else if (strides_y == 2 && strides_x == 2 && fy == 5 && fx == 5)
        details::convolve_go_template<2, 2, 5, 5>(filter, in_padded, out);

    else
        details::convolve_go(strides_y, strides_x, filter, in_padded, out);

    return out;
}

inline tensor3 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const tensor2& filter,
    const tensor3& in)
{
    const auto conv_func = [&](const tensor2& in_slice)
    {
        return convolve(out_height, out_width,
            strides_y, strides_x,
            padding_top, padding_bottom, padding_left, padding_right,
            filter, in_slice);
    };
    return
        tensor3_from_depth_slices(
            fplus::transform(
                conv_func,
                tensor3_to_depth_slices(in)));
}

inline tensor3 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const tensor3& filters,
    const tensor2& in)
{
    const auto conv_func = [&](const tensor2& filter_slice)
    {
        return convolve(out_height, out_width,
            strides_y, strides_x,
            padding_top, padding_bottom, padding_left, padding_right,
            filter_slice, in);
    };
    return
        tensor3_from_depth_slices(
            fplus::transform(
                conv_func,
                tensor3_to_depth_slices(filters)));
}

inline tensor2 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const tensor3& filter,
    const tensor3& in)
{
    const auto conv_func = [&](
        const tensor2& filter_slice, const tensor2& in_slice)
    {
        return convolve(out_height, out_width,
            strides_y, strides_x,
            padding_top, padding_bottom, padding_left, padding_right,
            filter_slice, in_slice);
    };
    assertion(filter.shape().depth_ == in.shape().depth_,
        "invalid filter depth");
    return
        sum_tensor2s(
            fplus::zip_with(
                conv_func,
                tensor3_to_depth_slices(filter),
                tensor3_to_depth_slices(in)));
}

inline tensor2 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const filter& f,
    const tensor3& in)
{
    const auto without_bias = convolve(out_height, out_width,
        strides_y, strides_x,
        padding_top, padding_bottom, padding_left, padding_right,
        f.get_tensor3(), in);
    const auto add_bias = [&](const float_t val)
    {
        return val + f.get_bias();
    };
    return transform_tensor2(add_bias, without_bias);
}

inline tensor3 convolve(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t padding_top,
    std::size_t padding_bottom,
    std::size_t padding_left,
    std::size_t padding_right,
    const std::vector<filter>& filters,
    const tensor3& in)
{
    // todo: convolve_matrix_mult instead of convolve with loops?
    //     (i.e. use im_to_col and matrix multiplication for performance)

    const auto convole_in = [&](const filter& f)
    {
        return convolve(out_height, out_width, strides_y, strides_x,
            padding_top, padding_bottom, padding_left, padding_right, f, in);
    };

    return tensor3_from_depth_slices(
        fplus::transform(convole_in, filters));
}

enum class padding { valid, same };

inline tensor3 convolve(
    shape2 strides,
    padding pad_type,
    const std::vector<filter>& filters,
    const tensor3& input)
{
    assertion(filters.size() > 0, "no filters");
    const auto filter_size = filters.front().shape();

    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_size.height_);
    const int filter_width = static_cast<int>(filter_size.width_);
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

    return convolve(
        static_cast<std::size_t>(out_height),
        static_cast<std::size_t>(out_width),
        static_cast<std::size_t>(strides_y),
        static_cast<std::size_t>(strides_x),
        static_cast<std::size_t>(pad_top),
        static_cast<std::size_t>(pad_bottom),
        static_cast<std::size_t>(pad_left),
        static_cast<std::size_t>(pad_right),
        filters, input);
}

} } // namespace fdeep, namespace internal
