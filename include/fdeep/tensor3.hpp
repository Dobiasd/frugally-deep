// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/tensor2.hpp"
#include "fdeep/tensor3_pos_yxz.hpp"
#include "fdeep/shape_hwc.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fdeep { namespace internal
{

class tensor3
{
public:
    tensor3(const shape_hwc& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.volume() == values->size(), "invalid number of values");
    }
    tensor3(const shape_hwc& shape, float_vec&& values) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(std::move(values)))
    {
        assertion(shape.volume() == values_->size(),
            "invalid number of values");
    }
    tensor3(const shape_hwc& shape, float_type value) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(shape.volume(), value))
    {
    }
    float_type get(const tensor3_pos_yxz& pos) const
    {
        return (*values_)[idx(pos)];
    }
    float_type get_yxz(std::size_t y, std::size_t x, std::size_t z) const
    {
        return get(tensor3_pos_yxz(y, x, z));
    }
    float_type get_x_y_padded(float_type pad_value,
        int y, int x, std::size_t z) const
    {
        if (y < 0 || y >= static_cast<int>(shape().height_) ||
            x < 0 || x >= static_cast<int>(shape().width_))
        {
            return pad_value;
        }
        return get(tensor3_pos_yxz(
            static_cast<std::size_t>(y), static_cast<std::size_t>(x), z));
    }
    void set(const tensor3_pos_yxz& pos, float_type value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set_yxz(std::size_t y, std::size_t x, std::size_t z, float_type value)
    {
        set(tensor3_pos_yxz(y, x, z), value);
    }
    const shape_hwc& shape() const
    {
        return shape_;
    }
    const shape_hw size_without_depth() const
    {
        return shape().without_depth();
    }
    std::size_t depth() const
    {
        return shape().depth_;
    }
    std::size_t height() const
    {
        return shape().height_;
    }
    std::size_t width() const
    {
        return shape().width_;
    }
    const shared_float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(const tensor3_pos_yxz& pos) const
    {
        return
            pos.y_ * shape().width_ * shape().depth_ +
            pos.x_ * shape().depth_ +
            pos.z_;
    };
    shape_hwc shape_;
    shared_float_vec values_;
};

typedef std::vector<tensor3> tensor3s;
typedef std::vector<tensor3s> tensor3s_vec;

template <typename F>
tensor3 transform_tensor3(F f, const tensor3& m)
{
    return tensor3(m.shape(), fplus::transform(f, *m.as_vector()));
}

inline tensor3 tensor3_from_depth_slices(const std::vector<tensor2>& ms)
{
    assertion(!ms.empty(), "no tensor2s");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor2, shape, shape_hw), ms),
        "all tensor2s must have the same size");
    std::size_t height = ms.front().shape().height_;
    std::size_t width = ms.front().shape().width_;
    tensor3 m(shape_hwc(height, width, ms.size()), 0);
    for (std::size_t z = 0; z < m.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < m.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < m.shape().width_; ++x)
            {
                m.set_yxz(y, x, z, ms[z].get_yx(y, x));
            }
        }
    }
    return m;
}

inline std::vector<tensor2> tensor3_to_tensor_2_depth_slices(const tensor3& m)
{
    std::vector<tensor2> ms;
    ms.reserve(m.shape().depth_);
    for (std::size_t i = 0; i < m.shape().depth_; ++i)
    {
        ms.push_back(tensor2(shape_hw(m.shape().height_, m.shape().width_), 0));
    }

    for (std::size_t z = 0; z < m.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < m.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < m.shape().width_; ++x)
            {
                ms[z].set_yx(y, x, m.get_yxz(y, x, z));
            }
        }
    }
    return ms;
}

inline tensor3 tensor2_to_tensor3(const tensor2& m)
{
    return tensor3(shape_hwc(m.shape().height_, m.shape().width_, 1),
        m.as_vector());
}

inline std::pair<tensor3_pos_yxz, tensor3_pos_yxz> tensor3_min_max_pos(
    const tensor3& vol)
{
    tensor3_pos_yxz result_min(0, 0, 0);
    tensor3_pos_yxz result_max(0, 0, 0);
    float_type value_max = std::numeric_limits<float_type>::lowest();
    float_type value_min = std::numeric_limits<float_type>::max();
    for (std::size_t z = 0; z < vol.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < vol.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < vol.shape().width_; ++x)
            {
                auto current_value = vol.get_yxz(y, x, z);
                if (current_value > value_max)
                {
                    result_max = tensor3_pos_yxz(y, x, z);
                    value_max = current_value;
                }
                if (current_value < value_min)
                {
                    result_min = tensor3_pos_yxz(y, x, z);
                    value_min = current_value;
                }
            }
        }
    }
    return std::make_pair(result_min, result_max);
}

inline tensor3_pos_yxz tensor3_max_pos(const tensor3& vol)
{
    return tensor3_min_max_pos(vol).second;
}

inline tensor3 tensor3_swap_depth_and_height(const tensor3& in)
{
    tensor3 result(shape_hwc(
        in.shape().depth_,
        in.shape().width_,
        in.shape().height_), 0);
    for (std::size_t z = 0; z < in.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                result.set_yxz(z, x, y, in.get_yxz(y, x, z));
            }
        }
    }
    return result;
}

inline tensor3 tensor3_swap_depth_and_width(const tensor3& in)
{
    tensor3 result(shape_hwc(
        in.shape().height_,
        in.shape().depth_,
        in.shape().width_), 0);
    for (std::size_t z = 0; z < in.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                result.set_yxz(y, z, x, in.get_yxz(y, x, z));
            }
        }
    }
    return result;
}

inline tensor3 concatenate_tensor3s_height(const tensor3s& ts)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(tensor3, width, std::size_t), ts),
        "all tensors must have the same width");

    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(tensor3, depth, std::size_t), ts),
        "all tensors must have the same depth");

    assertion(!ts.empty(), "no tensors to concatenate");

    const std::size_t height_sum = fplus::sum(fplus::transform(
        fplus_c_mem_fn_t(tensor3, height, std::size_t), ts));

    return tensor3(
        shape_hwc(
            height_sum,
            ts.front().shape().width_,
            ts.front().shape().depth_),
        fplus::transform_and_concat([](const tensor3& t) -> float_vec
        {
            return *t.as_vector();
        }, ts));
}

inline tensor3 concatenate_tensor3s_depth(const tensor3s& ts)
{
    return fplus::fwd::apply(ts,
        fplus::fwd::transform(tensor3_swap_depth_and_height),
        concatenate_tensor3s_height,
        tensor3_swap_depth_and_height);
}

inline tensor3 concatenate_tensor3s_width(const tensor3s& ts)
{
    return fplus::fwd::apply(ts,
        fplus::fwd::transform(tensor3_swap_depth_and_width),
        concatenate_tensor3s_depth,
        tensor3_swap_depth_and_width);
}

inline tensor3 concatenate_tensor3s(const tensor3s& ts, std::int32_t axis)
{
    if (axis == 1)
    {
        return concatenate_tensor3s_height(ts);
    }
    if (axis == 2)
    {
        return concatenate_tensor3s_width(ts);
    }
    assertion(axis == 0, "Invalid axis (" + std::to_string(axis) +
        ") for tensor concatenation.");
    return concatenate_tensor3s_depth(ts);
}

inline tensor3 flatten_tensor3(const tensor3& vol)
{
    float_vec values;
    values.reserve(vol.shape().volume());
    for (std::size_t y = 0; y < vol.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < vol.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < vol.shape().depth_; ++z)
            {
                values.push_back(vol.get_yxz(y, x, z));
            }
        }
    }
    return tensor3(shape_hwc(1, 1, values.size()), std::move(values));
}

inline tensor3 pad_tensor3(float_type val,
    std::size_t top_pad, std::size_t bottom_pad,
    std::size_t left_pad, std::size_t right_pad,
    const tensor3& in)
{
    tensor3 result(shape_hwc(
        in.shape().height_ + top_pad + bottom_pad,
        in.shape().width_ + left_pad + right_pad,
        in.shape().depth_), val);
    for (std::size_t z = 0; z < in.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                result.set_yxz(y + top_pad, x + left_pad, z, in.get_yxz(y, x, z));
            }
        }
    }
    return result;
}

inline tensor3 crop_tensor3(
    std::size_t top_crop, std::size_t bottom_crop,
    std::size_t left_crop, std::size_t right_crop,
    const tensor3& in)
{
    tensor3 result(shape_hwc(
        in.shape().height_ - (top_crop + bottom_crop),
        in.shape().width_ - (left_crop + right_crop),
        in.shape().depth_), 0);
    for (std::size_t z = 0; z < result.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < result.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < result.shape().width_; ++x)
            {
                result.set_yxz(y, x, z, in.get_yxz(y + top_crop, x + left_crop, z));
            }
        }
    }
    return result;
}

inline tensor3 dilate_tensor3(const shape_hw& dilation_rate, const tensor3& in)
{
    if (dilation_rate == shape_hw(1, 1))
    {
        return in;
    }

    tensor3 result(dilate_shape_hwc(dilation_rate, in.shape()), 0);
    for (std::size_t z = 0; z < in.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                result.set_yxz(
                    y * dilation_rate.height_,
                    x * dilation_rate.width_,
                    z,
                    in.get_yxz(y, x, z));
            }
        }
    }
    return result;
}

inline tensor3 reshape_tensor3(const tensor3& in,
    const std::vector<int>& target_shape)
{
    const auto shape = target_shape;
    assertion(shape.size() > 0 && shape.size() < 4,
        "Invalid shape in reshape");
    assertion(fplus::count(-1, shape) < 2,
        "Reshape can only infer one dimension");
    const auto fixed_dims = fplus::keep_if(fplus::is_not_equal_to(-1), shape);
    const auto fixes_dims_prod = fplus::product(fixed_dims);
    const auto num_values = static_cast<int>(in.as_vector()->size());
    assertion(num_values % fixes_dims_prod == 0,
        "Invalid dimensions in reshape");
    const auto deduced_dim = num_values / fixes_dims_prod;
    const auto deduced_shape = fplus::replace_elems(-1, deduced_dim, shape);
    assertion(fplus::product(deduced_shape) == num_values,
        "Invalid input tensor size in reshape");
    assertion(fplus::all_by(fplus::is_positive<int>, deduced_shape),
        "Invalid shape values in reshape");

    return tensor3(shape_hwc(
        static_cast<std::size_t>(deduced_shape[0]),
        static_cast<std::size_t>(deduced_shape[1]),
        static_cast<std::size_t>(deduced_shape[2])),
        in.as_vector());
}

inline tensor3 sum_tensor3s(const tensor3s& ts)
{
    assertion(!ts.empty(), "no tensor3s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor3, shape, shape_hwc), ts),
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

inline tensor3 multiply_tensor3s(const tensor3s& ts)
{
    assertion(!ts.empty(), "no tensor3s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor3, shape, shape_hwc), ts),
        "all tensor3s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor3, as_vector, shared_float_vec), ts);
    float_vec result_values;
    result_values.reserve(ts_values.front()->size());
    for (std::size_t i = 0; i < ts_values.front()->size(); ++i)
    {
        float_type product_val = static_cast<float_type>(1);
        for (const auto& t_vals : ts_values)
        {
            product_val *= (*t_vals)[i];
        }
        result_values.push_back(product_val);
    }
    return tensor3(ts.front().shape(), std::move(result_values));
}

inline tensor3 subtract_tensor3(const tensor3& a, const tensor3& b)
{
    assertion(a.shape() == b.shape(),
        "both tensor3s must have the same size");
    auto result_values = fplus::zip_with(std::minus<float_type>(),
        *a.as_vector(), *b.as_vector());
    return tensor3(a.shape(), std::move(result_values));
}

inline tensor3 average_tensor3s(const tensor3s& ts)
{
    const auto sum = sum_tensor3s(ts);
    const float_type divisor = static_cast<float_type>(ts.size());
    return transform_tensor3(fplus::multiply_with(1 / divisor), sum);
}

inline tensor3 max_tensor3s(const tensor3s& ts)
{
    assertion(!ts.empty(), "no tensor3s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor3, shape, shape_hwc), ts),
        "all tensor3s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor3, as_vector, shared_float_vec), ts);
    float_vec result_values;
    result_values.reserve(ts_values.front()->size());
    for (std::size_t i = 0; i < ts_values.front()->size(); ++i)
    {
        float_type max_val = std::numeric_limits<float_type>::lowest();
        for (const auto& t_vals : ts_values)
        {
            max_val = std::max(max_val, (*t_vals)[i]);
        }
        result_values.push_back(max_val);
    }
    return tensor3(ts.front().shape(), std::move(result_values));
}

} // namespace internal

using float_type = internal::float_type;
using float_vec = internal::float_vec;
using shared_float_vec = internal::shared_float_vec;
using tensor3 = internal::tensor3;
using tensor3s = internal::tensor3s;
using tensor3s_vec = internal::tensor3s_vec;


inline std::string show_tensor3(const tensor3& t)
{
    const auto xs = *t.as_vector();
    const auto test_strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', 0, 4), xs);
    const auto max_length = fplus::size_of_cont(fplus::maximum_on(
        fplus::size_of_cont<std::string>, test_strs));
    const auto strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', max_length, 4), xs);
    return fplus::show_cont(
        fplus::split_every(t.shape().height_,
            fplus::split_every(t.shape().width_, strs)));
}

inline std::string show_tensor3s(const tensor3s& ts)
{
    return fplus::show_cont(fplus::transform(show_tensor3, ts));
}

// Converts a memory block holding 8-bit values into a tensor3.
// Data must be stored row-wise (and channels_last).
// Scales the values from range [0, 255] into [low, high].
// May be used to convert an image (bgr, rgba, gray, etc.) to a tensor3.
inline tensor3 tensor3_from_bytes(const std::uint8_t* value_ptr,
    std::size_t height, std::size_t width, std::size_t channels,
    internal::float_type low = 0.0f, internal::float_type high = 1.0f)
{
    const std::vector<std::uint8_t> bytes(
        value_ptr, value_ptr + height * width * channels);
    auto values = fplus::transform_convert<float_vec>(
        [low, high](std::uint8_t b) -> internal::float_type
    {
        return fplus::reference_interval(low, high,
            static_cast<float_type>(0.0f),
            static_cast<float_type>(255.0f),
            static_cast<internal::float_type>(b));
    }, bytes);
    return tensor3(shape_hwc(height, width, channels), std::move(values));
}

// Converts a tensor3 into a memory block holding 8-bit values.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
// May be used to convert a tensor3 into an image.
inline void tensor3_into_bytes(const tensor3& t, std::uint8_t* value_ptr,
    std::size_t bytes_available,
    internal::float_type low = 0.0f, internal::float_type high = 1.0f)
{
    const auto values = t.as_vector();
    internal::assertion(bytes_available == values->size(),
    "invalid buffer size");
    const auto bytes = fplus::transform(
        [low, high](internal::float_type v) -> std::uint8_t
    {
        return fplus::round<internal::float_type, std::uint8_t>(
            fplus::reference_interval(
                static_cast<float_type>(0.0f),
                static_cast<float_type>(255.0f), low, high, v));
    }, *values);
    for (std::size_t i = 0; i < values->size(); ++i)
    {
        *(value_ptr++) = bytes[i];
    }
}

// Converts a tensor3 into a vector of bytes.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
inline std::vector<std::uint8_t> tensor3_to_bytes(const tensor3& t,
    internal::float_type low = 0.0f, internal::float_type high = 1.0f)
{
    std::vector<std::uint8_t> bytes(t.shape().volume(), 0);
    tensor3_into_bytes(t, bytes.data(), bytes.size(), low, high);
    return bytes;
}

inline tensor3s_vec reshape_tensor3_vectors(
    std::size_t vectors_size,
    std::size_t vector_size,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const tensor3s_vec& tss)
{
    const auto values = fplus::concat(fplus::concat(
        fplus::transform_inner(
            [](const tensor3& t) -> float_vec {return *t.as_vector();},
            tss)));

    fdeep::internal::assertion(values.size() == vectors_size * vector_size * height * width * depth,
        "Invalid number of values for reshape target.");

    const auto ts = fplus::transform(
        [&](fdeep::float_vec v) -> tensor3 {return tensor3(shape_hwc(height, width, depth), std::move(v));},
        fplus::split_every(depth * height * width, values));

    return fplus::split_every(vector_size, ts);
}

} // namespace fdeep
