#include "convolution.h"

#include <cassert>

matrix3d convolve_loops(const std::vector<filter>& filters,
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
                float val = 0.0f;
                const size3d& filt_size = filters[k].size();
                for (std::size_t z = 0; z < filt_size.depth(); ++z)
                {
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

matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol)
{
    // todo: convolve_matrix_mult
    //     use im_to_col and matrix multiplication for performance (?)
    return convolve_loops(filters, in_vol);
}