// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/common.hpp"

#include "frugally_deep/tensor2.hpp"
#include "frugally_deep/tensor3_pos.hpp"
#include "frugally_deep/shape3.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class tensor3
{
public:
    tensor3(const shape3& shape, const float_vec& values) :
        size_(shape),
        values_(values)
    {
        assertion(shape.volume() == values.size(), "invalid number of values");
    }
    explicit tensor3(const shape3& shape) :
        size_(shape),
        values_(shape.volume(), 0.0f)
    {
    }
    float_t get(const tensor3_pos& pos) const
    {
        return values_[idx(pos)];
    }
    float_t get(std::size_t z, std::size_t y, std::size_t x) const
    {
        return get(tensor3_pos(z, y, x));
    }
    void set(const tensor3_pos& pos, float_t value)
    {
        values_[idx(pos)] = value;
    }
    void set(std::size_t z, std::size_t y, std::size_t x, float_t value)
    {
        set(tensor3_pos(z, y, x), value);
    }
    const shape3& shape() const
    {
        return size_;
    }
    const shape2 size_without_depth() const
    {
        return shape().without_depth();
    }
    std::size_t depth() const
    {
        return shape().depth_;
    }
    const float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(const tensor3_pos& pos) const
    {
        return
            pos.z_ * shape().height_ * shape().width_ +
            pos.y_ * shape().width_ +
            pos.x_;
    };
    shape3 size_;
    float_vec values_;
};

typedef std::vector<tensor3> tensor3s;

inline std::string show_tensor3(const tensor3& m)
{
    std::string str;
    str += "[";
    for (std::size_t z = 0; z < m.shape().depth_; ++z)
    {
        str += "[";
        for (std::size_t y = 0; y < m.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < m.shape().width_; ++x)
            {
                str += fplus::show_float_fill_left<float_t>(' ', 8, 4,
                    m.get(z, y, x)) + ",";
            }
            str += "]\n";
        }
        str += "]\n";
    }
    str += "]";
    return str;
}

template <typename F>
tensor3 transform_tensor3(F f, const tensor3& m)
{
    return tensor3(m.shape(), fplus::transform(f, m.as_vector()));
}

inline tensor3 reshape_tensor3(const tensor3& in_vol, const shape3& out_size)
{
    return tensor3(out_size, in_vol.as_vector());
}

inline tensor2 depth_slice(std::size_t z, const tensor3& m)
{
    tensor2 result(shape2(m.shape().height_, m.shape().width_));
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            result.set(y, x, m.get(z, y, x));
        }
    }
    return result;
}

inline tensor3 tensor3_from_depth_slices(const std::vector<tensor2>& ms)
{
    assertion(!ms.empty(), "no tensor2s");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor2, shape, shape2), ms),
        "all tensor2s must have the same size");
    std::size_t height = ms.front().shape().height_;
    std::size_t width = ms.front().shape().width_;
    tensor3 m(shape3(ms.size(), height, width));
    for (std::size_t z = 0; z < m.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < m.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < m.shape().width_; ++x)
            {
                m.set(z, y, x, ms[z].get(y, x));
            }
        }
    }
    return m;
}

inline std::vector<tensor2> tensor3_to_depth_slices(const tensor3& m)
{
    std::vector<tensor2> ms(
        m.shape().depth_, tensor2(shape2(m.shape().height_, m.shape().width_)));

    for (std::size_t z = 0; z < m.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < m.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < m.shape().width_; ++x)
            {
                ms[z].set(y, x, m.get(z, y, x));
            }
        }
    }
    return ms;
}

inline tensor3 sparse_tensor3(std::size_t step, const tensor3& in)
{
    return tensor3_from_depth_slices(
        fplus::transform(
            fplus::bind_1st_of_2(sparse_tensor2, step),
            tensor3_to_depth_slices(in)));
}

inline tensor3 tensor2_to_tensor3(const tensor2& m)
{
    tensor3 result(shape3(1, m.shape().height_, m.shape().width_));
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            result.set(0, y, x, m.get(y, x));
        }
    }
    return result;
}

inline std::pair<tensor3_pos, tensor3_pos> tensor3_min_max_pos(
    const tensor3& vol)
{
    tensor3_pos result_min(0, 0, 0);
    tensor3_pos result_max(0, 0, 0);
    float_t value_max = std::numeric_limits<float_t>::lowest();
    float_t value_min = std::numeric_limits<float_t>::max();
    for (std::size_t z = 0; z < vol.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < vol.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < vol.shape().width_; ++x)
            {
                auto current_value = vol.get(z, y, x);
                if (current_value > value_max)
                {
                    result_max = tensor3_pos(z, y, x);
                    value_max = current_value;
                }
                if (current_value < value_min)
                {
                    result_min = tensor3_pos(z, y, x);
                    value_min = current_value;
                }
            }
        }
    }
    return std::make_pair(result_min, result_max);
}

inline tensor3_pos tensor3_max_pos(const tensor3& vol)
{
    return tensor3_min_max_pos(vol).second;
}

inline tensor3_pos tensor3_min_pos(const tensor3& vol)
{
    return tensor3_min_max_pos(vol).second;
}

inline std::pair<float_t, float_t> tensor3_min_max_value(const tensor3& vol)
{
    auto min_max_positions = tensor3_min_max_pos(vol);
    return std::make_pair(
        vol.get(min_max_positions.first), vol.get(min_max_positions.second));
}

inline float_t tensor3_max_value(const tensor3& m)
{
    return m.get(tensor3_max_pos(m));
}

inline float_t tensor3_min_value(const tensor3& m)
{
    return m.get(tensor3_min_pos(m));
}

inline tensor3 add_tensor3s(const tensor3& m1, const tensor3& m2)
{
    assertion(m1.shape() == m2.shape(), "unequal tensor shapes");
    return tensor3(m1.shape(), fplus::zip_with(std::plus<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

inline tensor3 concatenate_tensor3s(const tensor3s& ts)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(tensor3, size_without_depth, shape2), ts),
        "all tensors must have the same width and height");

    assertion(!ts.empty(), "no tensors to concatenate");

    const std::size_t depth_sum = fplus::sum(fplus::transform(
        fplus_c_mem_fn_t(tensor3, depth, std::size_t), ts));

    return tensor3(
        shape3(depth_sum, ts.front().shape().height_, ts.front().shape().width_),
        fplus::transform_and_concat(fplus_mem_fn(as_vector), ts));
}

inline tensor3 add_to_tensor3_elems(const tensor3& m, float_t x)
{
    return tensor3(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x + e;
    }, m.as_vector()));
}

inline tensor3 multiply_tensor3_elems(const tensor3& m, float_t x)
{
    return tensor3(m.shape(), fplus::transform([x](float_t e) -> float_t
    {
        return x * e;
    }, m.as_vector()));
}

inline tensor3 sum_tensor3s(const std::vector<tensor3>& ms)
{
    assertion(!ms.empty(), "no tensors given");
    return fplus::fold_left_1(add_tensor3s, ms);
}

inline tensor3 multiply_tensor3s_elementwise(
    const tensor3& m1, const tensor3& m2)
{
    assertion(m1.shape() == m2.shape(), "unequal tensor shapes");
    return tensor3(m1.shape(), fplus::zip_with(std::multiplies<float_t>(),
        m1.as_vector(), m2.as_vector()));
}

inline tensor3 multiply_tensor3(const tensor3& m, float_t factor)
{
    auto multiply_value_by_factor = [factor](const float_t x) -> float_t
    {
        return factor * x;
    };
    return transform_tensor3(multiply_value_by_factor, m);
}

inline tensor3 divide_tensor3(const tensor3& m, float_t divisor)
{
    return multiply_tensor3(m, 1 / divisor);
}

inline tensor3 mean_tensor3(const std::vector<tensor3>& ms)
{
    return divide_tensor3(sum_tensor3s(ms), static_cast<float_t>(ms.size()));
}

inline tensor3 sub_tensor3(const tensor3& m1, const tensor3& m2)
{
    return add_tensor3s(m1, multiply_tensor3(m2, -1));
}

inline tensor3 abs_tensor3_values(const tensor3& m)
{
    return transform_tensor3(fplus::abs<float_t>, m);
}

inline tensor3 abs_diff_tensor3s(const tensor3& m1, const tensor3& m2)
{
    return abs_tensor3_values(sub_tensor3(m1, m2));
}

inline float_t tensor3_sum_all_values(const tensor3& m)
{
    return fplus::sum(m.as_vector());
}

inline float_t tensor3_mean_value(const tensor3& m)
{
    return
        tensor3_sum_all_values(m) /
        static_cast<float_t>(m.shape().volume());
}

inline tensor3 operator + (const tensor3& lhs, const tensor3& rhs)
{
    return add_tensor3s(lhs, rhs);
}

inline tensor3 operator - (const tensor3& lhs, const tensor3& rhs)
{
    return sub_tensor3(lhs, rhs);
}

inline tensor3 operator * (const tensor3& m, float_t factor)
{
    return multiply_tensor3(m, factor);
}

inline tensor3 operator / (const tensor3& m, float_t divisor)
{
    return divide_tensor3(m, divisor);
}

inline bool operator == (const tensor3& a, const tensor3& b)
{
    return a.shape() == b.shape() && a.as_vector() == b.as_vector();
}

inline bool operator != (const tensor3& a, const tensor3& b)
{
    return !(a == b);
}

inline tensor3& operator += (tensor3& lhs, const tensor3& rhs)
{
    lhs = lhs + rhs;
    return lhs;
}

inline tensor3 transpose_tensor3(const tensor3& m)
{
    return
        tensor3_from_depth_slices(
            fplus::transform(
                transpose_tensor2,
                tensor3_to_depth_slices(m)));
}

inline tensor3 flip_tensor3_horizontally(const tensor3& m)
{
    return
        tensor3_from_depth_slices(
            fplus::transform(
                flip_tensor2_horizontally,
                tensor3_to_depth_slices(m)));
}

inline tensor3 rotate_tensor3_ccw(int step_cnt_90_deg, const tensor3& m)
{
    return
        tensor3_from_depth_slices(
            fplus::transform(
                fplus::bind_1st_of_2(rotate_tensor2_ccw, step_cnt_90_deg),
                tensor3_to_depth_slices(m)));
}

inline tensor3 flatten_tensor3(const tensor3& vol)
{
    float_vec values;
    values.reserve(vol.shape().volume());
    for (std::size_t x = 0; x < vol.shape().width_; ++x)
    {
        for (std::size_t y = 0; y < vol.shape().height_; ++y)
        {
            for (std::size_t z = 0; z < vol.shape().depth_; ++z)
            {
                values.push_back(vol.get(z, y, x));
            }
        }
    }
    return tensor3(shape3(1, 1, values.size()), values);
}

} } // namespace fdeep, namespace internal
