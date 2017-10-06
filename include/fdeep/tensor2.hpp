// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2.hpp"
#include "fdeep/tensor2_pos.hpp"

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
    tensor2(const shape2& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.area() == values->size(), "invalid number of values");
    }
    tensor2(const shape2& shape, float_vec&& values) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(std::move(values)))
    {
        assertion(shape.area() == values_->size(), "invalid number of values");
    }
    tensor2(const shape2& shape, float_t value) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(shape.area(), value))
    {
    }

    const float_t& get(const tensor2_pos& pos) const
    {
        return (*values_)[idx(pos)];
    }
    const float_t& get(std::size_t y, std::size_t x) const
    {
        return get(tensor2_pos(y, x));
    }
    float_t& get(const tensor2_pos& pos)
    {
        return (*values_)[idx(pos)];
    }
    float_t& get(std::size_t y, std::size_t x)
    {
        return get(tensor2_pos(y, x));
    }
    void set(const tensor2_pos& pos, float_t value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set(std::size_t y, std::size_t x, float_t value)
    {
        set(tensor2_pos(y, x), value);
    }
    const shape2& shape() const
    {
        return shape_;
    }
    const shared_float_vec& as_vector() const
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
    shared_float_vec values_;
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

inline tensor2 multiply(const tensor2& a, const tensor2& b)
{
    assertion(a.shape().width_ == b.shape().height_, "invalid tensor shapes");

    tensor2 m(shape2(a.shape().height_, b.shape().width_),
        static_cast<float_t>(0.0f));

    for (std::size_t i = 0; i < a.shape().height_; ++i)
    {
        for (std::size_t k = 0; k < a.shape().width_; ++k)
        {
            float_t a_i_k = a.get(i, k);
            for (std::size_t j = 0; j < b.shape().width_; ++j)
            {
                m.get(i, j) += a_i_k * b.get(k, j);
            }
        }
    }
    return m;
}

inline float_t tensor2_sum_all_values(const tensor2& m)
{
    return fplus::sum(*m.as_vector());
}

inline float_t tensor2_mean_value(const tensor2& m)
{
    return
        tensor2_sum_all_values(m) /
        static_cast<float_t>(m.shape().area());
}

inline tensor2 add_to_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x + e;
    }, *m.as_vector()));
}

inline tensor2 sub_from_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return e - x;
    }, *m.as_vector()));
}

inline tensor2 multiply_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x * e;
    }, *m.as_vector()));
}

inline tensor2 divide_tensor2_elems(const tensor2& m, float_t x)
{
    return tensor2(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return e / x;
    }, *m.as_vector()));
}

} } // namespace fdeep, namespace internal
