// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/size2d.h"
#include "frugally_deep/matrix2d_pos.h"

#include <fplus/fplus.hpp>

#include <cassert>
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
    float_t get(const matrix2d_pos& pos) const
    {
        return values_[idx(pos)];
    }
    float_t get(std::size_t y, std::size_t x) const
    {
        return get(matrix2d_pos(y, x));
    }
    void set(const matrix2d_pos& pos, float_t value)
    {
        values_[idx(pos)] = value;
    }
    void set(std::size_t y, std::size_t x, float_t value)
    {
        set(matrix2d_pos(y, x), value);
    }
    const size2d& size() const
    {
        return size_;
    }
    const float_vec& as_vector() const
    {
        return values_;
    }
    void overwrite_values(const float_vec_const_it vs_begin,
        const float_vec_const_it vs_end)
    {
        assert(static_cast<std::size_t>(std::distance(vs_begin, vs_end)) ==
            size().area());
        std::copy(vs_begin, vs_end, std::begin(values_));
    }

private:
    std::size_t idx(const matrix2d_pos& pos) const
    {
        return
            pos.y_ * size().width_ +
            pos.x_;
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
    // todo: use as_vector instead to avoid nested loops
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

inline matrix2d reshape_matrix2d(const matrix2d& m, const size2d& out_size)
{
    return matrix2d(out_size, m.as_vector());
}

inline matrix2d sparse_matrix2d(std::size_t step, const matrix2d& in)
{
    matrix2d out(size2d(
        in.size().height_ * step - (step - 1),
        in.size().width_ * step - (step - 1)));
    for (std::size_t y = 0; y < in.size().height_; ++y)
    {
        for (std::size_t x = 0; x < in.size().width_; ++x)
        {
            out.set(y * step, x * step, in.get(y, x));
        }
    }
    return out;
}

inline matrix2d multiply(const matrix2d& a, const matrix2d& b)
{
    assert(a.size().width_ == b.size().height_);

    std::size_t inner = a.size().width_;
    matrix2d m(size2d(a.size().height_, b.size().width_));

    for (std::size_t y = 0; y < a.size().height_; ++y)
    {
        for (std::size_t x = 0; x < b.size().width_; ++x)
        {
            float_t sum = 0;
            for (std::size_t i = 0; i < inner; ++i)
            {
                sum += a.get(y, i) * b.get(i, x);
            }
            m.set(y, x, sum);
        }
    }
    return m;
}

inline float_t matrix2d_sum_all_values(const matrix2d& m)
{
    return fplus::sum(m.as_vector());
}

inline matrix2d add_matrix2ds(const matrix2d& m1, const matrix2d& m2)
{
    assert(m1.size() == m2.size());
    return matrix2d(m1.size(), fplus::zip_with(std::plus<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

inline matrix2d sum_matrix2ds(const std::vector<matrix2d>& ms)
{
    assert(!ms.empty());
    return fplus::fold_left_1(add_matrix2ds, ms);
}

inline matrix2d transpose_matrix2d(const matrix2d& m)
{
    matrix2d result(size2d(m.size().width_, m.size().height_));
    for (std::size_t x = 0; x < m.size().width_; ++x)
    {
        for (std::size_t y = 0; y < m.size().height_; ++y)
        {
            result.set(x, y, m.get(y, x));
        }
    }
    return result;
}

inline matrix2d flip_matrix2d_horizontally(const matrix2d& m)
{
    matrix2d result(m.size());
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            result.set(y, m.size().width_ - (x + 1), m.get(y, x));
        }
    }
    return result;
}

inline matrix2d rotate_matrix2d_ccw(int step_cnt_90_deg, const matrix2d& m)
{
    if (step_cnt_90_deg < 0)
    {
        step_cnt_90_deg = 4 - ((-step_cnt_90_deg) % 4);
    }
    step_cnt_90_deg = step_cnt_90_deg % 4;
    if (step_cnt_90_deg == 0)
    {
        return m;
    }
    else if (step_cnt_90_deg == 1)
    {
        return transpose_matrix2d(flip_matrix2d_horizontally(m));
    }
    else if (step_cnt_90_deg == 2)
    {
        return rotate_matrix2d_ccw(1, rotate_matrix2d_ccw(1, m));
    }
    else if (step_cnt_90_deg == 3)
    {
        return flip_matrix2d_horizontally(transpose_matrix2d(m));
    }
    else
    {
        assert(false);
        return m;
    }
}

} // namespace fd
