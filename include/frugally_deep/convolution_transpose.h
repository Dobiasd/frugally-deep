// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/filter.h"
#include "frugally_deep/convolution.h"

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

namespace internal
{
    inline void convolve_transpose_go(
        std::size_t stride,
        const matrix2d& filter,
        const matrix2d& in,
        matrix2d& out)
    {
        const std::size_t fy = filter.size().height_;
        const std::size_t fx = filter.size().width_;
        for (std::size_t y = 0; y < in.size().height_; ++y)
        {
            for (std::size_t x = 0; x < in.size().width_; ++x)
            {
                for (std::size_t yf = 0; yf < fy; ++yf)
                {
                    for (std::size_t xf = 0; xf < fx; ++xf)
                    {
                        const float_t add_val = filter.get(yf, xf) *
                            in.get(y + yf, x + xf);
                        out.set(stride * y, stride * x, out.get(y, x) + add_val);
                    }
                }
            }
        }
    }


    template <
        std::size_t stride,
        std::size_t fy,
        std::size_t fx
        >
    void convolve_transpose_go_template(
        const matrix2d& filter,
        const matrix2d& in,
        matrix2d& out)
    {
        assert(filter.size().height_ == fy);
        assert(filter.size().width_ == fx);
        for (std::size_t y = 0; y < in.size().height_; ++y)
        {
            for (std::size_t x = 0; x < in.size().width_; ++x)
            {
                for (std::size_t yf = 0; yf < fy; ++yf)
                {
                    for (std::size_t xf = 0; xf < fx; ++xf)
                    {
                        const float_t add_val = filter.get(yf, xf) *
                            in.get(y + yf, x + xf);
                        out.set(stride * y, stride * x, out.get(y, x) + add_val);
                    }
                }
            }
        }
    }
} // namespace internal

// todo: noch ist alles falsch
/*
0xxxx0
fff
 fff
  fff
   fff
 yyyy


0xxxx0
ffff
  ffff
 yy
 */
inline matrix2d convolve_transpose(
    std::size_t stride,
    const matrix2d& filter,
    const matrix2d& in)
{
    assert(stride > 0);
    const std::size_t h1 = in.size().height_;
    const std::size_t w1 = in.size().width_;
    const std::size_t fy = filter.size().height_;
    const std::size_t fx = filter.size().width_;
    const std::size_t h2 = (fx - stride) * h1;
    const std::size_t w2 = (fx - stride) * w1;

    matrix2d out(size2d(h2, w2));

    if (stride == 1 && fy == 1 && fx == 1)
        internal::convolve_transpose_go_template<1, 1, 1>(filter, in, out);

    else if (stride == 1 && fy == 3 && fx == 3)
        internal::convolve_transpose_go_template<1, 3, 3>(filter, in, out);
    else if (stride == 1 && fy == 5 && fx == 5)
        internal::convolve_transpose_go_template<1, 5, 5>(filter, in, out);

    else if (stride == 2 && fy == 1 && fx == 1)
        internal::convolve_transpose_go_template<2, 1, 1>(filter, in, out);

    else if (stride == 2 && fy == 3 && fx == 3)
        internal::convolve_transpose_go_template<2, 3, 3>(filter, in, out);
    else if (stride == 2 && fy == 5 && fx == 5)
        internal::convolve_transpose_go_template<2, 5, 5>(filter, in, out);

    else
        internal::convolve_transpose_go(stride, filter, in, out);

    return out;
}

inline matrix3d convolve_transpose(
    std::size_t stride,
    const matrix2d& filter,
    const matrix3d& in)
{
    const auto conv_transpose_func = [&](const matrix2d& in_slice)
    {
        return convolve_transpose(
            stride, filter, in_slice);
    };
    return
        matrix3d_from_depth_slices(
            fplus::transform(
                conv_transpose_func,
                matrix3d_to_depth_slices(in)));
}

inline matrix2d convolve_transpose(
    std::size_t stride,
    const matrix3d& filter,
    const matrix3d& in)
{
    const auto conv_transpose_func = [&](
        const matrix2d& filter_slice, const matrix2d& in_slice)
    {
        return convolve_transpose(
            stride, filter_slice, in_slice);
    };
    return
        sum_matrix2ds(
            fplus::zip_with(
                conv_transpose_func,
                matrix3d_to_depth_slices(filter),
                matrix3d_to_depth_slices(in)));
}

inline matrix2d convolve_transpose(
    std::size_t stride,
    const filter& f,
    const matrix3d& in)
{
    const auto without_bias = convolve_transpose(
        stride, f.get_matrix3d(), in);
    const auto add_bias = [&](const float_t val)
    {
        return val + f.get_bias();
    };
    return transform_matrix2d(add_bias, without_bias);
}

inline matrix3d convolve_transpose(
    std::size_t stride,
    const std::vector<filter>& filters,
    const matrix3d& in)
{
    // todo: convolve_transpose_matrix_mult instead of convolve with loops?
    //     (i.e. use im_to_col and matrix multiplication for performance)

    const auto convolve_transpose_in = [&](const filter& f)
    {
        return convolve_transpose(stride, f, in);
    };

    return matrix3d_from_depth_slices(
        fplus::transform(convolve_transpose_in, filters));
}

} // namespace fd
