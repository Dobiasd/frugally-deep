// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/size3d.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fd
{

class matrix3d
{
public:
    explicit matrix3d(const size3d& size, const float_vec& values) :
        size_(size),
        values_(values)
    {
        assert(size.volume() == values.size());
    }
    explicit matrix3d(const size3d& size) :
        size_(size),
        values_(size.volume(), 0.0f)
    {
    }
    float_t get(std::size_t z, std::size_t y, std::size_t x) const
    {
        return values_[idx(z, y, x)];
    }
    void set(std::size_t z, std::size_t y, size_t x, float_t value)
    {
        values_[idx(z, y, x)] = value;
    }
    const size3d& size() const
    {
        return size_;
    }
    const float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(std::size_t z, std::size_t y, size_t x) const
    {
        return z * size().height() * size().width() + y * size().width() + x;
    };
    size3d size_;
    float_vec values_;
};

inline std::string show_matrix3d(const matrix3d& m)
{
    std::string str;
    str += "[";
    for (std::size_t z = 0; z < m.size().depth(); ++z)
    {
        str += "[";
        for (std::size_t y = 0; y < m.size().height(); ++y)
        {
            for (std::size_t x = 0; x < m.size().width(); ++x)
            {
                str += std::to_string(m.get(z, y, x)) + ",";
            }
            str += "]\n";
        }
        str += "]\n";
    }
    str += "]";
    return str;
}

template <typename F>
matrix3d transform_matrix3d(F f, const matrix3d& in_vol)
{
    matrix3d out_vol(size3d(
        in_vol.size().depth(),
        in_vol.size().height(),
        in_vol.size().width()));
    for (std::size_t z = 0; z < in_vol.size().depth(); ++z)
    {
        for (std::size_t y = 0; y < in_vol.size().height(); ++y)
        {
            for (std::size_t x = 0; x < in_vol.size().width(); ++x)
            {
                out_vol.set(z, y, x, f(in_vol.get(z, y, x)));
            }
        }
    }
    return out_vol;
}

matrix3d reshape_matrix3d(const matrix3d& in_vol, const size3d& out_size)
{
    return matrix3d(out_size, in_vol.as_vector());
}

} // namespace fd
