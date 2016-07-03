#include "filter.h"

#include <cassert>

std::vector<float> filter::get_params() const
{
    std::vector<float> params;
    params.reserve(param_count());
    for (std::size_t z = 0; z < m_.size().depth(); ++z)
    {
        for (std::size_t y = 0; y < m_.size().height(); ++y)
        {
            for (std::size_t x = 0; x < m_.size().width(); ++x)
            {
                params.push_back(m_.get(z, y, x));
            }
        }
    }
    return params;
}

void filter::set_params(const std::vector<float>& params)
{
    assert(params.size() == param_count());
    std::size_t i = 0;
    for (std::size_t z = 0; z < m_.size().depth(); ++z)
    {
        for (std::size_t y = 0; y < m_.size().height(); ++y)
        {
            for (std::size_t x = 0; x < m_.size().width(); ++x)
            {
                m_.set(z, y, x, params[++i]);
            }
        }
    }
}