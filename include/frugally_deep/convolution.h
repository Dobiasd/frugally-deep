// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/filter.h"

#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep
{

inline tensor2 pad_tensor2(
    std::size_t padding_y,
    std::size_t padding_x,
    const tensor2& in)
{
    tensor2 out(shape2(
        in.size().height_ + 2 * padding_y,
        in.size().width_ + 2 * padding_x));
    for (std::size_t y = 0; y < in.size().height_; ++y)
    {
        for (std::size_t x = 0; x < in.size().width_; ++x)
        {
            out.set(y + padding_y, x + padding_x, in.get(y, x));
        }
    }
    return out;
}

namespace internal
{
    inline void convolve_go(
        std::size_t stride,
        const tensor2& filter,
        const tensor2& in,
        tensor2& out)
    {
        const std::size_t fy = filter.size().height_;
        const std::size_t fx = filter.size().width_;
        for (std::size_t y = 0; y < out.size().height_; ++y)
        {
            for (std::size_t x = 0; x < out.size().width_; ++x)
            {
                for (std::size_t yf = 0; yf < fy; ++yf)
                {
                    for (std::size_t xf = 0; xf < fx; ++xf)
                    {
                        const float_t add_val = filter.get(yf, xf) *
                            in.get(stride * y + yf, stride * x + xf);
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
        std::size_t stride,
        std::size_t fy,
        std::size_t fx
        >
    void convolve_go_template(
        const tensor2& filter,
        const tensor2& in,
        tensor2& out)
    {
        assert(filter.size().height_ == fy);
        assert(filter.size().width_ == fx);
        for (std::size_t y = 0; y < out.size().height_; ++y)
        {
            for (std::size_t x = 0; x < out.size().width_; ++x)
            {
                for (std::size_t yf = 0; yf < fy; ++yf)
                {
                    for (std::size_t xf = 0; xf < fx; ++xf)
                    {
                        const float_t add_val = filter.get(yf, xf) *
                            in.get(stride * y + yf, stride * x + xf);
                        out.set(y, x, out.get(y, x) + add_val);
                    }
                }
            }
        }
    }
} // namespace internal

inline tensor2 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const tensor2& filter,
    const tensor2& in_orig)
{
    assert(stride > 0);
    const std::size_t h1 = in_orig.size().height_;
    const std::size_t w1 = in_orig.size().width_;
    const std::size_t fy = filter.size().height_;
    const std::size_t fx = filter.size().width_;
    const std::size_t py = padding_y;
    const std::size_t px = padding_x;

    assert(fy <= h1);
    assert(fx <= w1);

    const std::size_t h2 = (h1 - fy + 2 * py) / stride + 1;
    const std::size_t w2 = (w1 - fx + 2 * px) / stride + 1;

    const tensor2 in_padded = pad_tensor2(padding_y, padding_x, in_orig);

    tensor2 out(shape2(h2, w2));

    if (stride == 1 && fy == 1 && fx == 1)
        internal::convolve_go_template<1, 1, 1>(filter, in_padded, out);

    else if (stride == 1 && fy == 3 && fx == 3)
        internal::convolve_go_template<1, 3, 3>(filter, in_padded, out);
    else if (stride == 1 && fy == 5 && fx == 5)
        internal::convolve_go_template<1, 5, 5>(filter, in_padded, out);

    else if (stride == 2 && fy == 1 && fx == 1)
        internal::convolve_go_template<2, 1, 1>(filter, in_padded, out);

    else if (stride == 2 && fy == 3 && fx == 3)
        internal::convolve_go_template<2, 3, 3>(filter, in_padded, out);
    else if (stride == 2 && fy == 5 && fx == 5)
        internal::convolve_go_template<2, 5, 5>(filter, in_padded, out);

    else
        internal::convolve_go(stride, filter, in_padded, out);

    return out;
}

inline tensor3 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const tensor2& filter,
    const tensor3& in)
{
    const auto conv_func = [&](const tensor2& in_slice)
    {
        return convolve(
            stride, padding_y, padding_x, filter, in_slice);
    };
    return
        tensor3_from_depth_slices(
            fplus::transform(
                conv_func,
                tensor3_to_depth_slices(in)));
}

inline tensor3 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const tensor3& filters,
    const tensor2& in)
{
    const auto conv_func = [&](const tensor2& filter_slice)
    {
        return convolve(
            stride, padding_y, padding_x, filter_slice, in);
    };
    return
        tensor3_from_depth_slices(
            fplus::transform(
                conv_func,
                tensor3_to_depth_slices(filters)));
}

inline tensor2 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const tensor3& filter,
    const tensor3& in)
{
    const auto conv_func = [&](
        const tensor2& filter_slice, const tensor2& in_slice)
    {
        return convolve(
            stride, padding_y, padding_x, filter_slice, in_slice);
    };
    return
        sum_tensor2s(
            fplus::zip_with(
                conv_func,
                tensor3_to_depth_slices(filter),
                tensor3_to_depth_slices(in)));
}

inline tensor2 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const filter& f,
    const tensor3& in)
{
    const auto without_bias = convolve(
        stride, padding_y, padding_x, f.get_tensor3(), in);
    const auto add_bias = [&](const float_t val)
    {
        return val + f.get_bias();
    };
    return transform_tensor2(add_bias, without_bias);
}

inline tensor3 convolve(
    std::size_t stride,
    std::size_t padding_y,
    std::size_t padding_x,
    const std::vector<filter>& filters,
    const tensor3& in)
{
    // todo: convolve_matrix_mult instead of convolve with loops?
    //     (i.e. use im_to_col and matrix multiplication for performance)

    const auto convole_in = [&](const filter& f)
    {
        return convolve(stride, padding_y, padding_x, f, in);
    };

    return tensor3_from_depth_slices(
        fplus::transform(convole_in, filters));
}

} // namespace fdeep
