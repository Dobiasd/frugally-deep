// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/common.hpp"

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
        shape_(shape),
        values_(values)
    {
        assertion(shape.area() == values.size(), "invalid number of values");
    }
    tensor2(const shape2& shape, const float_t& value) :
        shape_(shape),
        values_(fplus::replicate(shape.area(), value))
    {
    }
    explicit tensor2(const shape2& shape) :
        shape_(shape),
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
    const shape2& shape() const
    {
        return shape_;
    }
    const float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(const tensor2_pos& pos) const
    {
        return
            pos.y_ * shape().width_ +
            pos.x_;
    };
    shape2 shape_;
    float_vec values_;
};

inline std::string show_tensor2(const tensor2& m)
{
    std::string str;
    str += "[";
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
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
    return tensor2(m.shape(), fplus::transform(f, m.as_vector()));
}

inline tensor2 reshape_tensor2(const tensor2& m, const shape2& out_shape)
{
    return tensor2(out_shape, m.as_vector());
}

inline tensor2 sparse_tensor2(std::size_t step, const tensor2& in)
{
    tensor2 out(shape2(
        in.shape().height_ * step - (step - 1),
        in.shape().width_ * step - (step - 1)));
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            out.set(y * step, x * step, in.get(y, x));
        }
    }
    return out;
}

inline tensor2 multiply(const tensor2& a, const tensor2& b)
{
    assertion(a.shape().width_ == b.shape().height_, "invalid tensor shapes");

    std::size_t inner = a.shape().width_;
    tensor2 m(shape2(a.shape().height_, b.shape().width_));

    for (std::size_t y = 0; y < a.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < b.shape().width_; ++x)
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
        static_cast<float_t>(m.shape().area());
}

inline tensor2 add_tensor2s(const tensor2& m1, const tensor2& m2)
{
    assertion(m1.shape() == m2.shape(), "unequal tensor shapes");
    return tensor2(m1.shape(), fplus::zip_with(std::plus<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

inline tensor2 add_to_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x + e;
    }, m.as_vector()));
}

inline tensor2 sub_from_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return e - x;
    }, m.as_vector()));
}

inline tensor2 multiply_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x * e;
    }, m.as_vector()));
}

inline tensor2 divide_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return e / x;
    }, m.as_vector()));
}

inline tensor2 sum_tensor2s(const std::vector<tensor2>& ms)
{
    assertion(!ms.empty(), "no tensors given");
    return fplus::fold_left_1(add_tensor2s, ms);
}

inline tensor2 flip_tensor2_horizontally(const tensor2& m)
{
    tensor2 result(m.shape());
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            result.set(y, m.shape().width_ - (x + 1), m.get(y, x));
        }
    }
    return result;
}

} } // namespace fdeep, namespace internal
