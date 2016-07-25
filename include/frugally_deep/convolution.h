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

namespace fd
{

// todo: convolution for even and odd filter kernel sizes in both dims

namespace internal
{
    bool are_sizes_allowed_to_convolute(
        const size2d& filter_size,
        const size2d& data_size)
    {
        return filter_size.height_ <= data_size.height_ &&
            filter_size.width_ <= data_size.width_;
    }

    bool are_sizes_allowed_to_convolute(
        const size3d& filter_size,
        const size3d& data_size)
    {
        return filter_size.depth_ == data_size.depth_ &&
            are_sizes_allowed_to_convolute(
                filter_size.without_depth(), data_size.without_depth());
    }

    // Give the compiler a chance to unroll the inner loops.
    // In tests with a 3x3 filter and clang++ -O3
    //     the performance was increased by a factor of two by this.
    // todo: profiling:
    //       is idx-calculation in filters[k].get a performance bottleneck?
    template <std::size_t filter_height, std::size_t filter_width>
    inline matrix2d convolve_loops_fixed_filter_size(
        const matrix3d& filter,
        const matrix3d& in_vol)
    {
        const size3d& filt_size = filter.size();
        assert(filter_height == filt_size.height_);
        assert(filter_width == filt_size.width_);
        assert(are_sizes_allowed_to_convolute(filter.size().without_depth(), in_vol.size().without_depth()));
        matrix2d out(size2d(
            in_vol.size().height_ + 1 - filter_height,
            in_vol.size().width_ + 1 - filter_width));
        for (std::size_t y = 0; y < out.size().height_; ++y)
        {
            for (std::size_t x = 0; x < out.size().width_; ++x)
            {
                float_t val = 0;
                for (std::size_t z = 0; z < filt_size.depth_; ++z)
                {
                    for (std::size_t yf = 0; yf < filter_height; ++yf)
                    {
                        for (std::size_t xf = 0; xf < filter_width; ++xf)
                        {
                            val += filter.get(z, yf, xf) *
                                in_vol.get(z, y + yf, x + xf);
                        }
                    }
                }
                out.set(y, x, val);
            }
        }
        return out;
    }

    inline matrix2d convolve_loops(const matrix3d& filter,
        const matrix3d& in_vol)
    {
        const size3d& filt_size = filter.size();
        assert(are_sizes_allowed_to_convolute(filter.size(), in_vol.size()));
        if (filt_size.height_ == 1 && filt_size.width_ == 1)
            return convolve_loops_fixed_filter_size<1, 1>(filter, in_vol);
        if (filt_size.height_ == 1 && filt_size.width_ == 3)
            return convolve_loops_fixed_filter_size<1, 3>(filter, in_vol);
        if (filt_size.height_ == 3 && filt_size.width_ == 1)
            return convolve_loops_fixed_filter_size<3, 1>(filter, in_vol);
        if (filt_size.height_ == 3 && filt_size.width_ == 3)
            return convolve_loops_fixed_filter_size<3, 3>(filter, in_vol);
        if (filt_size.height_ == 1 && filt_size.width_ == 5)
            return convolve_loops_fixed_filter_size<1, 5>(filter, in_vol);
        if (filt_size.height_ == 5 && filt_size.width_ == 1)
            return convolve_loops_fixed_filter_size<5, 1>(filter, in_vol);
        if (filt_size.height_ == 5 && filt_size.width_ == 5)
            return convolve_loops_fixed_filter_size<5, 5>(filter, in_vol);

        const size_t filter_height = filt_size.height_;
        const size_t filter_width = filt_size.width_;
        matrix2d out(size2d(
            in_vol.size().height_ + 1 - filter_height,
            in_vol.size().width_ + 1 - filter_width));
        for (std::size_t y = 0; y < out.size().height_; ++y)
        {
            for (std::size_t x = 0; x < out.size().width_; ++x)
            {
                float_t val = 0;
                for (std::size_t z = 0; z < filt_size.depth_; ++z)
                {
                    for (std::size_t yf = 0; yf < filter_height; ++yf)
                    {
                        for (std::size_t xf = 0; xf < filter_width; ++xf)
                        {
                            val += filter.get(z, yf, xf) *
                                in_vol.get(z, y + yf, x + xf);
                        }
                    }
                }
                out.set(y, x, val);
            }
        }
        return out;
    }

    // todo: auch fuer performance templaten
    inline matrix3d convolve_loops(const matrix2d& filter,
        const matrix3d& in_vol)
    {
        const size2d& filt_size = filter.size();
        assert(are_sizes_allowed_to_convolute(filter.size(), in_vol.size().without_depth()));

        const size_t filter_height = filt_size.height_;
        const size_t filter_width = filt_size.width_;
        matrix3d out(size3d(
            in_vol.size().depth_,
            in_vol.size().height_ + 1 - filter_height,
            in_vol.size().width_ + 1 - filter_width));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < out.size().height_; ++y)
            {
                for (std::size_t x = 0; x < out.size().width_; ++x)
                {
                    float_t val = 0;
                    for (std::size_t yf = 0; yf < filter_height; ++yf)
                    {
                        for (std::size_t xf = 0; xf < filter_width; ++xf)
                        {
                            val += filter.get(yf, xf) *
                                in_vol.get(z, y + yf, x + xf);
                        }
                    }
                    out.set(z, y, x, val);
                }
            }
        }
        return out;
    }
}

inline matrix2d convolve(const matrix3d& filter, const matrix3d& in_vol)
{
    return internal::convolve_loops(filter, in_vol);
}

inline matrix3d convolve(const matrix2d& filter, const matrix3d& in_vol)
{
    return internal::convolve_loops(filter, in_vol);
}

inline matrix2d convolve(const filter& f, const matrix3d& in_vol)
{
    const auto without_bias = convolve(f.get_matrix3d(), in_vol);
    const auto add_bias = [&](const float_t val)
    {
        return val + f.get_bias();
    };
    return transform_matrix2d(add_bias, without_bias);
}

inline matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol)
{
    // todo: convolve_matrix_mult instead of convolve_loops?
    //     (use im_to_col and matrix multiplication for performance)

    const auto convole_in_vol = [&](const filter& f)
    {
        return convolve(f, in_vol);
    };

    return matrix3d_from_depth_slices(
        fplus::transform(convole_in_vol, filters));
}

} // namespace fd
