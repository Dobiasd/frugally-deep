// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/size2d.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fd
{

class matrix2d
{
public:
    matrix2d(const size2d& shape, const float_vec& values) :
        size_(shape),
        values_(values)
    {
        assert(shape.area() == values.size());
    }
    explicit matrix2d(const size2d& shape) :
        size_(shape),
        values_(shape.area(), 0.0f)
    {
    }
    float_t get(std::size_t y, size_t x) const
    {
        return values_[idx(y, x)];
    }
    void set(std::size_t y, size_t x, float_t value)
    {
        values_[idx(y, x)] = value;
    }
    const size2d& size() const
    {
        return size_;
    }
    const float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(std::size_t y, std::size_t x) const
    {
        return y * size().width_ + x;
    };
    size2d size_;
    float_vec values_;
};

inline std::string show_matrix2d(const matrix2d& m)
{
    std::string str;
    str += "[";
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            str += std::to_string(m.get(y, x)) + ",";
        }
        str += "]\n";
    }
    str += "]";
    return str;
}

template <typename F>
matrix2d transform_matrix2d(F f, const matrix2d& m)
{
    matrix2d out_vol(size2d(
        m.size().height_,
        m.size().width_));
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            out_vol.set(y, x, f(m.get(y, x)));
        }
    }
    return out_vol;
}

matrix2d reshape_matrix2d(const matrix2d& m, const size2d& out_size)
{
    return matrix2d(out_size, m.as_vector());
}

} // namespace fd
