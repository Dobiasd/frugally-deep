// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d_pos.h"
#include "frugally_deep/size3d.h"

#include "fplus/fplus.h"

#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace fd
{

class matrix3d
{
public:
    matrix3d(const size3d& size, const float_vec& values) :
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
    // todo: use as_vector instead to avoid nested loops
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

matrix3d_pos matrix3d_max_pos(const matrix3d& vol)
{
    matrix3d_pos result(0, 0, 0);
    float_t value = std::numeric_limits<float_t>::min();
    for (std::size_t z = 0; z < vol.size().depth(); ++z)
    {
        for (std::size_t y = 0; y < vol.size().height(); ++y)
        {
            for (std::size_t x = 0; x < vol.size().width(); ++x)
            {
                auto current_value = vol.get(z, y, x);
                if (current_value > value)
                {
                    result = matrix3d_pos(z, y, x);
                    value = current_value;
                }
            }
        }
    }
    return result;
}

matrix3d add_matrix3ds(const matrix3d& m1, const matrix3d& m2)
{
    assert(m1.size() == m2.size());
    return matrix3d(m1.size(), fplus::zip_with(std::plus<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

matrix3d multiply_matrix3d(const matrix3d& m, float_t factor)
{
    auto multiply_value_by_factor = [factor](const float_t x) -> float_t
    {
        return factor * x;
    };
    return transform_matrix3d(multiply_value_by_factor, m);
}

matrix3d divide_matrix3d(const matrix3d& m, float_t divisor)
{
    return multiply_matrix3d(m, 1 / divisor);
}

matrix3d sub_matrix3d(const matrix3d& m1, const matrix3d& m2)
{
    return add_matrix3ds(m1, multiply_matrix3d(m2, -1));
}

matrix3d abs_matrix3d_values(const matrix3d& m)
{
    return transform_matrix3d(fplus::abs<float_t>, m);
}

matrix3d abs_diff_matrix3ds(const matrix3d& m1, const matrix3d& m2)
{
    return abs_matrix3d_values(sub_matrix3d(m1, m2));
}

float_t matrix3d_sum_all_values(const matrix3d& m)
{
    return fplus::sum(m.as_vector());
}

matrix3d operator + (const matrix3d& lhs, const matrix3d& rhs)
{
    return add_matrix3ds(lhs, rhs);
}

matrix3d operator - (const matrix3d& lhs, const matrix3d& rhs)
{
    return sub_matrix3d(lhs, rhs);
}

matrix3d operator * (const matrix3d& m, float_t factor)
{
    return multiply_matrix3d(m, factor);
}

matrix3d operator / (const matrix3d& m, float_t divisor)
{
    return divide_matrix3d(m, divisor);
}

} // namespace fd
