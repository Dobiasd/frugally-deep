// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.hpp"

#include "frugally_deep/shape2.hpp"
#include "frugally_deep/tensor2_pos.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class tensor2
{
public:
    tensor2(const shape2& shape, const float_vec& values) :
        size_(shape),
        values_(values)
    {
        assert(shape.area() == values.size());
    }
    tensor2(const shape2& shape, const float_t& value) :
        size_(shape),
        values_(fplus::replicate(shape.area(), value))
    {
    }
    explicit tensor2(const shape2& shape) :
        size_(shape),
        values_(shape.area(), 0.0f)
    {
    }
    float_t get(const tensor2_pos& pos) const
    {
        return values_[idx(pos)];
    }
    float_t get(std::size_t y, std::size_t x) const
    {
        return get(tensor2_pos(y, x));
    }
    void set(const tensor2_pos& pos, float_t value)
    {
        values_[idx(pos)] = value;
    }
    void set(std::size_t y, std::size_t x, float_t value)
    {
        set(tensor2_pos(y, x), value);
    }
    const shape2& size() const
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
    std::size_t idx(const tensor2_pos& pos) const
    {
        return
            pos.y_ * size().width_ +
            pos.x_;
    };
    shape2 size_;
    float_vec values_;
};

inline std::string show_tensor2(const tensor2& m)
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
tensor2 transform_tensor2(F f, const tensor2& m)
{
    return tensor2(m.size(), fplus::transform(f, m.as_vector()));
}

inline tensor2 reshape_tensor2(const tensor2& m, const shape2& out_size)
{
    return tensor2(out_size, m.as_vector());
}

inline tensor2 sparse_tensor2(std::size_t step, const tensor2& in)
{
    tensor2 out(shape2(
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

inline tensor2 multiply(const tensor2& a, const tensor2& b)
{
    assert(a.size().width_ == b.size().height_);

    std::size_t inner = a.size().width_;
    tensor2 m(shape2(a.size().height_, b.size().width_));

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

inline float_t tensor2_sum_all_values(const tensor2& m)
{
    return fplus::sum(m.as_vector());
}

inline float_t tensor2_mean_value(const tensor2& m)
{
    return
        tensor2_sum_all_values(m) /
        static_cast<float_t>(m.size().area());
}

inline tensor2 add_tensor2s(const tensor2& m1, const tensor2& m2)
{
    assert(m1.size() == m2.size());
    return tensor2(m1.size(), fplus::zip_with(std::plus<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

inline tensor2 add_to_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.size(), fplus::transform([x](float_t e) -> float_t
    {
        return x + e;
    }, m.as_vector()));
}

inline tensor2 sub_from_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.size(), fplus::transform([x](float_t e) -> float_t
    {
        return e - x;
    }, m.as_vector()));
}

inline tensor2 multiply_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.size(), fplus::transform([x](float_t e) -> float_t
    {
        return x * e;
    }, m.as_vector()));
}

inline tensor2 divide_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.size(), fplus::transform([x](float_t e) -> float_t
    {
        return e / x;
    }, m.as_vector()));
}

inline tensor2 sum_tensor2s(const std::vector<tensor2>& ms)
{
    assert(!ms.empty());
    return fplus::fold_left_1(add_tensor2s, ms);
}

inline tensor2 transpose_tensor2(const tensor2& m)
{
    tensor2 result(shape2(m.size().width_, m.size().height_));
    for (std::size_t x = 0; x < m.size().width_; ++x)
    {
        for (std::size_t y = 0; y < m.size().height_; ++y)
        {
            result.set(x, y, m.get(y, x));
        }
    }
    return result;
}

inline tensor2 flip_tensor2_horizontally(const tensor2& m)
{
    tensor2 result(m.size());
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            result.set(y, m.size().width_ - (x + 1), m.get(y, x));
        }
    }
    return result;
}

inline tensor2 rotate_tensor2_ccw(int step_cnt_90_deg, const tensor2& m)
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
        return transpose_tensor2(flip_tensor2_horizontally(m));
    }
    else if (step_cnt_90_deg == 2)
    {
        return rotate_tensor2_ccw(1, rotate_tensor2_ccw(1, m));
    }
    else if (step_cnt_90_deg == 3)
    {
        return flip_tensor2_horizontally(transpose_tensor2(m));
    }
    else
    {
        assert(false);
        return m;
    }
}

} } // namespace fdeep, namespace internal
