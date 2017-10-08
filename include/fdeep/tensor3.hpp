// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/tensor2.hpp"
#include "fdeep/tensor3_pos.hpp"
#include "fdeep/shape3.hpp"

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
    tensor3(const shape3& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.volume() == values->size(), "invalid number of values");
    }
    tensor3(const shape3& shape, float_vec&& values) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(std::move(values)))
    {
        assertion(shape.volume() == values_->size(),
            "invalid number of values");
    }
    tensor3(const shape3& shape, float_type value) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(shape.volume(), value))
    {
    }
    float_type get(const tensor3_pos& pos) const
    {
        return (*values_)[idx(pos)];
    }
    float_type get(std::size_t z, std::size_t y, std::size_t x) const
    {
        return get(tensor3_pos(z, y, x));
    }
    void set(const tensor3_pos& pos, float_type value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set(std::size_t z, std::size_t y, std::size_t x, float_type value)
    {
        set(tensor3_pos(z, y, x), value);
    }
    const shape3& shape() const
    {
        return shape_;
    }
    const shape2 size_without_depth() const
    {
        return shape().without_depth();
    }
    std::size_t depth() const
    {
        return shape().depth_;
    }
    const shared_float_vec& as_vector() const
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
    shape3 shape_;
    shared_float_vec values_;
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
                str += fplus::show_float_fill_left<float_type>(' ', 8, 4,
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
    return tensor3(m.shape(), fplus::transform(f, *m.as_vector()));
}

inline tensor2 depth_slice(std::size_t z, const tensor3& m)
{
    tensor2 result(shape2(m.shape().height_, m.shape().width_), 0);
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
    tensor3 m(shape3(ms.size(), height, width), 0);
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
    std::vector<tensor2> ms;
    ms.reserve(m.shape().depth_);
    for (std::size_t i = 0; i < m.shape().depth_; ++i)
    {
        ms.push_back(tensor2(shape2(m.shape().height_, m.shape().width_), 0));
    }

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

inline tensor3 tensor2_to_tensor3(const tensor2& m)
{
    return tensor3(shape3(1, m.shape().height_, m.shape().width_),
        m.as_vector());
}

inline std::pair<tensor3_pos, tensor3_pos> tensor3_min_max_pos(
    const tensor3& vol)
{
    tensor3_pos result_min(0, 0, 0);
    tensor3_pos result_max(0, 0, 0);
    float_type value_max = std::numeric_limits<float_type>::lowest();
    float_type value_min = std::numeric_limits<float_type>::max();
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

inline float_type tensor3_max_value(const tensor3& m)
{
    return m.get(tensor3_max_pos(m));
}

inline float_type tensor3_min_value(const tensor3& m)
{
    return m.get(tensor3_min_pos(m));
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
        shape3(depth_sum,
            ts.front().shape().height_, ts.front().shape().width_),
        fplus::transform_and_concat([](const tensor3& t) -> float_vec
        {
            return *t.as_vector();
        }, ts));
}

inline tensor3 add_to_tensor3_elems(const tensor3& m, float_type x)
{
    return tensor3(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return x + e;
    }, *m.as_vector()));
}

inline tensor3 multiply_tensor3_elems(const tensor3& m, float_type x)
{
    return tensor3(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return x * e;
    }, *m.as_vector()));
}

inline tensor3 multiply_tensor3s_elementwise(
    const tensor3& m1, const tensor3& m2)
{
    assertion(m1.shape() == m2.shape(), "unequal tensor shapes");
    return tensor3(m1.shape(), fplus::zip_with(std::multiplies<float_type>(),
        *m1.as_vector(), *m2.as_vector()));
}

inline tensor3 multiply_tensor3(const tensor3& m, float_type factor)
{
    auto multiply_value_by_factor = [factor](const float_type x) -> float_type
    {
        return factor * x;
    };
    return transform_tensor3(multiply_value_by_factor, m);
}

inline tensor3 divide_tensor3(const tensor3& m, float_type divisor)
{
    return multiply_tensor3(m, 1 / divisor);
}

inline tensor3 abs_tensor3_values(const tensor3& m)
{
    return transform_tensor3(fplus::abs<float_type>, m);
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
    return tensor3(shape3(values.size(), 1, 1), std::move(values));
}

inline tensor3 pad_tensor3(float_type val,
    std::size_t top_pad, std::size_t bottom_pad,
    std::size_t left_pad, std::size_t right_pad,
    const tensor3& in)
{
    tensor3 result(shape3(
        in.shape().depth_,
        in.shape().height_ + top_pad + bottom_pad,
        in.shape().width_ + left_pad + right_pad), val);
    for (std::size_t z = 0; z < in.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                result.set(z, y + top_pad, x + left_pad, in.get(z, y, x));
            }
        }
    }
    return result;
}

// (height, width, depth) -> (depth, height, width)
inline tensor3 depth_last_to_depth_first(const tensor3& in)
{
    tensor3 result(shape3(
        in.shape().width_,
        in.shape().depth_,
        in.shape().height_), 0);
    for (std::size_t x = 0; x < in.shape().width_; ++x)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(x, z, y, in.get(z, y, x));
            }
        }
    }
    return result;
}

// (depth, height, width) -> (height, width, depth)
inline tensor3 depth_first_to_depth_last(const tensor3& in)
{
    tensor3 result(shape3(
        in.shape().height_,
        in.shape().width_,
        in.shape().depth_), 0);
    for (std::size_t x = 0; x < in.shape().width_; ++x)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(y, x, z, in.get(z, y, x));
            }
        }
    }
    return result;
}

inline tensor3 sum_tensor3s(const tensor3s& ts)
{
    assertion(!ts.empty(), "no tensor3s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor3, shape, shape3), ts),
        "all tensor3s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor3, as_vector, shared_float_vec), ts);
    float_vec result_values;
    result_values.reserve(ts_values.front()->size());
    for (std::size_t i = 0; i < ts_values.front()->size(); ++i)
    {
        float_type sum_val = static_cast<float_type>(0);
        for (const auto& t_vals : ts_values)
        {
            sum_val += (*t_vals)[i];
        }
        result_values.push_back(sum_val);
    }
    return tensor3(ts.front().shape(), std::move(result_values));
}

} // namespace internal

using float_type = internal::float_type;
using tensor3 = internal::tensor3;
using tensor3s = internal::tensor3s;

// assumes pixels in 8-bit BGR format, data stored row-wise
inline tensor3 tensor3_from_bgr_image(const std::uint8_t* value_ptr,
    std::size_t height, std::size_t width)
{
    const std::vector<std::uint8_t> bytes(
        value_ptr, value_ptr + height * width * 3);
    auto values = fplus::transform([](std::uint8_t b) -> internal::float_type
    {
        return static_cast<internal::float_type>(b) / 255;
    }, bytes);
    return internal::depth_last_to_depth_first(
        tensor3(shape3(height, width, 3), std::move(values)));
}

// converts a tensor to a 8-bit BGR image, data stored row-wise
inline void tensor3_to_bgr_image(const tensor3& t, std::uint8_t* value_ptr,
    std::size_t bytes_available)
{
    const auto values = depth_first_to_depth_last(t).as_vector();
    internal::assertion(bytes_available == values->size(),
    "invalid buffer size");
    const auto bytes = fplus::transform(
        [](internal::float_type v) -> std::uint8_t
    {
        return static_cast<std::uint8_t>(
            fplus::clamp<internal::float_type>(0, 255, v * 255));
    }, *values);
    std::copy(std::begin(bytes), std::end(bytes), value_ptr);
}

} // namespace fdeep
