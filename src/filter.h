#pragma once

#include "typedefs.h"

#include "matrix3d.h"
#include "size3d.h"

#include <cstddef>
#include <vector>

class filter
{
public:
    explicit filter(const matrix3d& m) : m_(m)
    {
    }
    std::size_t param_count() const
    {
        return m_.size().area();
    }
    const size3d& size() const
    {
        return m_.size();
    }
    matrix3d get_matrix3d() const
    {
        return m_;
    }
    float_t get(std::size_t z, std::size_t y, size_t x) const
    {
        return m_.get(z, y, x);
    }
    float_vec get_params() const
    {
        float_vec params;
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
    void set_params(const float_vec& params)
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
private:
    matrix3d m_;
};
