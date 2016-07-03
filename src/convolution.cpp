#include "convolution.h"

#include <cassert>

matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol)
{
    // todo: padding
    assert(in_vol.size().depth() == filters.size());
    matrix3d out_vol(size3d(
        filters.size(),
        in_vol.size().height(),
        in_vol.size().width()));
    // todo: use im_to_col and matrix multiplication for performance (?)
    for (std::size_t k = 0; k < filters.size(); ++k)
    {
        for (std::size_t y = 1; y < in_vol.size().height() - 1; ++y)
        {
            for (std::size_t x = 1; x < in_vol.size().width() - 1; ++x)
            {
                float val = 0.0f;
                for (std::size_t z = 0; z < filters[k].size().depth(); ++z)
                {
                    for (std::size_t yf = 0; yf < filters[k].size().height(); ++yf)
                    {
                        for (std::size_t xf = 0; xf < filters[k].size().width(); ++xf)
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