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

namespace internal
{
    inline matrix3d convolve_loops(const std::vector<filter>& filters,
        const matrix3d& in_vol)
    {
        // todo: padding
        assert(in_vol.size().depth() == filters.size());
        matrix3d out_vol(size3d(
            filters.size(),
            in_vol.size().height(),
            in_vol.size().width()));
        for (std::size_t k = 0; k < filters.size(); ++k)
        {
            for (std::size_t y = 1; y < in_vol.size().height() - 1; ++y)
            {
                for (std::size_t x = 1; x < in_vol.size().width() - 1; ++x)
                {
                    float_t val = 0;
                    const size3d& filt_size = filters[k].size();
                    for (std::size_t z = 0; z < filt_size.depth(); ++z)
                    {
                        // todo: performance optimization:
                        // special versions for filters of size 3*3, 5*5 etc
                        for (std::size_t yf = 0; yf < filt_size.height(); ++yf)
                        {
                            for (std::size_t xf = 0; xf < filt_size.width(); ++xf)
                            {
                                val += filters[k].get(z, yf, xf) *
                                    in_vol.get(z, y - 1 + yf, x - 1 + xf);
                            }
                        }
                    }
                    out_vol.set(k, y, x, val);
                }
            }
        }
        return out_vol;
    }
}

inline matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol)
{
    // todo: convolve_matrix_mult instead of convolve_loops
    //     use im_to_col and matrix multiplication for performance (?)
    return internal::convolve_loops(filters, in_vol);
}

} // namespace fd
