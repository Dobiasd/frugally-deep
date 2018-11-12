// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/tensor5_pos.hpp"
#include "fdeep/shape5.hpp"

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

class tensor5
{
public:
    tensor5(const shape5& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.volume() == values->size(), "invalid number of values");
    }
    tensor5(const shape5& shape, float_vec&& values) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(std::move(values)))
    {
        assertion(shape.volume() == values_->size(),
            "invalid number of values");
    }
    tensor5(const shape5& shape, float_type value) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(shape.volume(), value))
    {
    }
    float_type get(const tensor5_pos& pos) const
    {
        return (*values_)[idx(pos)];
    }
    float_type get(std::size_t pos_dim_5, std::size_t pos_dim_4,
        std::size_t y, std::size_t x, std::size_t z) const
    {
        return get(tensor5_pos(pos_dim_5, pos_dim_4, y, x, z));
    }
    float_type get_x_y_padded(float_type pad_value,
        int y, int x, std::size_t z) const
    {
        if (y < 0 || y >= static_cast<int>(shape().height_) ||
            x < 0 || x >= static_cast<int>(shape().width_))
        {
            return pad_value;
        }
        return get(tensor5_pos(0, 0,
            static_cast<std::size_t>(y), static_cast<std::size_t>(x), z));
    }
    void set(const tensor5_pos& pos, float_type value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set(std::size_t pos_dim_5, std::size_t pos_dim_4,
        std::size_t y, std::size_t x, std::size_t z, float_type value)
    {
        set(tensor5_pos(pos_dim_5, pos_dim_4, y, x, z), value);
    }
    const shape5& shape() const
    {
        return shape_;
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
    std::size_t idx(const tensor5_pos& pos) const
    {
        return
            pos.pos_dim_5_ * shape().size_dim_4_ * shape().height_ * shape().width_ * shape().depth_ +
            pos.pos_dim_4_ * shape().height_ * shape().width_ * shape().depth_ +
            pos.y_ * shape().width_ * shape().depth_ +
            pos.x_ * shape().depth_ +
            pos.z_;
    };
    shape5 shape_;
    shared_float_vec values_;
};

typedef std::vector<tensor5> tensor5s;
typedef std::vector<tensor5s> tensor5s_vec;

template <typename F>
tensor5 transform_tensor5(F f, const tensor5& m)
{
    return tensor5(m.shape(), fplus::transform(f, *m.as_vector()));
}

inline tensor5 tensor5_from_depth_slices(const std::vector<tensor5>& ms)
{
    assertion(!ms.empty(), "no slices given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor5, shape, shape5), ms),
        "all slices must have the same size");
    for (const auto& m : ms)
    {
        m.shape().assert_is_shape_2();
    }
    std::size_t height = ms.front().shape().height_;
    std::size_t width = ms.front().shape().width_;
    tensor5 m(shape5(1, 1, height, width, ms.size()), 0);
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < m.shape().depth_; ++z)
            {
                m.set(0, 0, y, x, z, ms[z].get(0, 0, y, x, 0));
            }
        }
    }
    return m;
}

inline std::vector<tensor5> tensor5_to_depth_slices(const tensor5& m)
{
    std::vector<tensor5> ms;
    ms.reserve(m.shape().depth_);
    for (std::size_t i = 0; i < m.shape().depth_; ++i)
    {
        ms.push_back(tensor5(shape5(1, 1,
            m.shape().height_, m.shape().width_, 1), 0));
    }

    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < m.shape().depth_; ++z)
            {
                ms[z].set(0, 0, y, x, 0, m.get(0, 0, y, x, z));
            }
        }
    }
    return ms;
}

inline tensor5s tensor5_to_tensor5s_width_slices(const tensor5& m)
{
    tensor5s ms;
    ms.reserve(m.shape().width_);
    for (std::size_t i = 0; i < m.shape().width_; ++i)
    {
        ms.push_back(tensor5(shape5(m.shape().size_dim_5_,
                                    m.shape().size_dim_4_,
                                    m.shape().height_,
                                    1,
                                    m.shape().depth_), 0));
    }
    for (std::size_t dim5 = 0; dim5 < m.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < m.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < m.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < m.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < m.shape().depth_; ++z)
                    {
                        ms[x].set(tensor5_pos(dim5, dim4, y, 0, z), m.get(tensor5_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensor5s tensor5_to_tensor5s_height_slices(const tensor5& m)
{
    tensor5s ms;
    ms.reserve(m.shape().height_);
    for (std::size_t i = 0; i < m.shape().height_; ++i)
    {
        ms.push_back(tensor5(shape5(m.shape().size_dim_5_,
                                    m.shape().size_dim_4_,
                                    1,
                                    m.shape().width_,
                                    m.shape().depth_), 0));
    }
    for (std::size_t dim5 = 0; dim5 < m.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < m.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < m.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < m.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < m.shape().depth_; ++z)
                    {
                        ms[y].set(tensor5_pos(dim5, dim4, 0, x, z), m.get(tensor5_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensor5s tensor5_to_tensor5s_dim4_slices(const tensor5& m)
{
    tensor5s ms;
    ms.reserve(m.shape().size_dim_4_);
    for (std::size_t i = 0; i < m.shape().size_dim_4_; ++i)
    {
        ms.push_back(tensor5(shape5(m.shape().size_dim_5_,
                                    1,
                                    m.shape().height_,
                                    m.shape().width_,
                                    m.shape().depth_), 0));
    }
    for (std::size_t dim5 = 0; dim5 < m.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < m.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < m.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < m.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < m.shape().depth_; ++z)
                    {
                        ms[dim4].set(tensor5_pos(dim5, 0, y, x, z), m.get(tensor5_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensor5s tensor5_to_tensor5s_dim5_slices(const tensor5& m)
{
    tensor5s ms;
    ms.reserve(m.shape().size_dim_5_);
    for (std::size_t i = 0; i < m.shape().size_dim_5_; ++i)
    {
        ms.push_back(tensor5(shape5(1,
                                    m.shape().size_dim_4_,
                                    m.shape().height_,
                                    m.shape().width_,
                                    m.shape().depth_), 0));
    }
    for (std::size_t dim5 = 0; dim5 < m.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < m.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < m.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < m.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < m.shape().depth_; ++z)
                    {
                        ms[dim5].set(tensor5_pos(0, dim4, y, x, z), m.get(tensor5_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline std::pair<tensor5_pos, tensor5_pos> tensor5_min_max_pos(
    const tensor5& vol)
{
    tensor5_pos result_min(0, 0, 0, 0, 0);
    tensor5_pos result_max(0, 0, 0, 0, 0);
    float_type value_max = std::numeric_limits<float_type>::lowest();
    float_type value_min = std::numeric_limits<float_type>::max();
    for (std::size_t y = 0; y < vol.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < vol.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < vol.shape().depth_; ++z)
            {
                auto current_value = vol.get(0, 0, y, x, z);
                if (current_value > value_max)
                {
                    result_max = tensor5_pos(0, 0, y, x, z);
                    value_max = current_value;
                }
                if (current_value < value_min)
                {
                    result_min = tensor5_pos(0, 0, y, x, z);
                    value_min = current_value;
                }
            }
        }
    }
    return std::make_pair(result_min, result_max);
}

inline tensor5_pos tensor5_max_pos(const tensor5& vol)
{
    return tensor5_min_max_pos(vol).second;
}

inline tensor5 tensor5_swap_depth_and_height(const tensor5& in)
{
    tensor5 result(shape5(1, 1,
        in.shape().depth_,
        in.shape().width_,
        in.shape().height_), 0);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(0, 0, z, x, y, in.get(0, 0, y, x, z));
            }
        }
    }
    return result;
}

inline tensor5 tensor5_swap_depth_and_width(const tensor5& in)
{
    tensor5 result(shape5(1, 1,
        in.shape().height_,
        in.shape().depth_,
        in.shape().width_), 0);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(0, 0, y, z, x, in.get(0, 0, y, x, z));
            }
        }
    }
    return result;
}

inline tensor5 concatenate_tensor5s_dim4(const tensor5s& in)
{
    tensor5 result(shape5(in.front().shape().size_dim_5_,
                          in.size(),
                          in.front().shape().height_,
                          in.front().shape().width_,
                          in.front().shape().depth_), 0);
    for (std::size_t dim5 = 0; dim5 < in.front().shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < in.size(); ++dim4)
        {
            for (std::size_t y = 0; y < in.front().shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.front().shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < in.front().shape().depth_; ++z)
                    {
                        result.set(tensor5_pos(dim5, dim4, y, x, z), in[dim4].get(tensor5_pos(dim5, 0, y, x, z)));
                    }
                }
            }
        }
    }
    return result;
}

inline tensor5 concatenate_tensor5s_dim5(const tensor5s& in)
{
    tensor5 result(shape5(in.size(),
                          in.front().shape().size_dim_4_,
                          in.front().shape().height_,
                          in.front().shape().width_,
                          in.front().shape().depth_), 0);
    for (std::size_t dim5 = 0; dim5 < in.size(); ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < in.front().shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < in.front().shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.front().shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < in.front().shape().depth_; ++z)
                    {
                        result.set(tensor5_pos(dim5, dim4, y, x, z), in[dim5].get(tensor5_pos(0, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return result;
}

inline tensor5 concatenate_tensor5s_height(const tensor5s& ts)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(tensor5, width, std::size_t), ts),
        "all tensors must have the same width");

    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(tensor5, depth, std::size_t), ts),
        "all tensors must have the same depth");

    assertion(!ts.empty(), "no tensors to concatenate");

    const std::size_t height_sum = fplus::sum(fplus::transform(
        fplus_c_mem_fn_t(tensor5, height, std::size_t), ts));

    return tensor5(
        shape5(1, 1,
            height_sum,
            ts.front().shape().width_,
            ts.front().shape().depth_),
        fplus::transform_and_concat([](const tensor5& t) -> float_vec
        {
            return *t.as_vector();
        }, ts));
}

inline tensor5 concatenate_tensor5s_depth(const tensor5s& ts)
{
    return fplus::fwd::apply(ts,
        fplus::fwd::transform(tensor5_swap_depth_and_height),
        concatenate_tensor5s_height,
        tensor5_swap_depth_and_height);
}

inline tensor5 concatenate_tensor5s_width(const tensor5s& ts)
{
    return fplus::fwd::apply(ts,
        fplus::fwd::transform(tensor5_swap_depth_and_width),
        concatenate_tensor5s_depth,
        tensor5_swap_depth_and_width);
}

inline tensor5 concatenate_tensor5s(const tensor5s& ts, std::int32_t axis)
{
    if (axis == 1)
    {
        return concatenate_tensor5s_height(ts);
    }
    if (axis == 2)
    {
        return concatenate_tensor5s_width(ts);
    }
    if (axis == 3)
    {
        return concatenate_tensor5s_dim4(ts);
    }
    if (axis == 4)
    {
        return concatenate_tensor5s_dim5(ts);
    }
    assertion(axis == 0, "Invalid axis (" + std::to_string(axis) +
        ") for tensor concatenation.");
    return concatenate_tensor5s_depth(ts);
}

inline tensor5 flatten_tensor5(const tensor5& vol)
{
    return tensor5(shape5(1, 1, 1, 1, vol.shape().volume()), vol.as_vector());
}

inline tensor5 pad_tensor5(float_type val,
    std::size_t top_pad, std::size_t bottom_pad,
    std::size_t left_pad, std::size_t right_pad,
    const tensor5& in)
{
    tensor5 result(shape5(1, 1,
        in.shape().height_ + top_pad + bottom_pad,
        in.shape().width_ + left_pad + right_pad,
        in.shape().depth_), val);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(0, 0, y + top_pad, x + left_pad, z, in.get(0, 0, y, x, z));
            }
        }
    }
    return result;
}

inline tensor5 crop_tensor5(
    std::size_t top_crop, std::size_t bottom_crop,
    std::size_t left_crop, std::size_t right_crop,
    const tensor5& in)
{
    tensor5 result(shape5(1, 1,
        in.shape().height_ - (top_crop + bottom_crop),
        in.shape().width_ - (left_crop + right_crop),
        in.shape().depth_), 0);
    for (std::size_t y = 0; y < result.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < result.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < result.shape().depth_; ++z)
            {
                result.set(0, 0, y, x, z, in.get(0, 0, y + top_crop, x + left_crop, z));
            }
        }
    }
    return result;
}

inline tensor5 dilate_tensor5(const shape2& dilation_rate, const tensor5& in)
{
    if (dilation_rate == shape2(1, 1))
    {
        return in;
    }

    tensor5 result(dilate_shape5(dilation_rate, in.shape()), 0);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(0, 0,
                    y * dilation_rate.height_,
                    x * dilation_rate.width_,
                    z,
                    in.get(0, 0, y, x, z));
            }
        }
    }
    return result;
}

inline tensor5 reshape_tensor5(const tensor5& in,
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

    return tensor5(shape5(1, 1,
        static_cast<std::size_t>(deduced_shape[0]),
        static_cast<std::size_t>(deduced_shape[1]),
        static_cast<std::size_t>(deduced_shape[2])),
        in.as_vector());
}

inline tensor5 sum_tensor5s(const tensor5s& ts)
{
    assertion(!ts.empty(), "no tensor5s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor5, shape, shape5), ts),
        "all tensor5s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor5, as_vector, shared_float_vec), ts);
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
    return tensor5(ts.front().shape(), std::move(result_values));
}

inline tensor5 multiply_tensor5s(const tensor5s& ts)
{
    assertion(!ts.empty(), "no tensor5s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor5, shape, shape5), ts),
        "all tensor5s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor5, as_vector, shared_float_vec), ts);
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
    return tensor5(ts.front().shape(), std::move(result_values));
}

inline tensor5 subtract_tensor5(const tensor5& a, const tensor5& b)
{
    assertion(a.shape() == b.shape(),
        "both tensor5s must have the same size");
    auto result_values = fplus::zip_with(std::minus<float_type>(),
        *a.as_vector(), *b.as_vector());
    return tensor5(a.shape(), std::move(result_values));
}

inline tensor5 average_tensor5s(const tensor5s& ts)
{
    const auto sum = sum_tensor5s(ts);
    const float_type divisor = static_cast<float_type>(ts.size());
    return transform_tensor5(fplus::multiply_with(1 / divisor), sum);
}

inline tensor5 max_tensor5s(const tensor5s& ts)
{
    assertion(!ts.empty(), "no tensor5s given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor5, shape, shape5), ts),
        "all tensor5s must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor5, as_vector, shared_float_vec), ts);
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
    return tensor5(ts.front().shape(), std::move(result_values));
}

inline RowMajorMatrixXf eigen_row_major_mat_from_values(std::size_t height,
    std::size_t width, const float_vec& values)
{
    assertion(height * width == values.size(), "invalid shape");
    RowMajorMatrixXf m(height, width);
    std::memcpy(m.data(), values.data(), values.size() * sizeof(float_type));
    return m;
}

inline shared_float_vec eigen_row_major_mat_to_values(const RowMajorMatrixXf& m)
{
    shared_float_vec result = fplus::make_shared_ref<float_vec>();
    result->resize(static_cast<std::size_t>(m.rows() * m.cols()));
    std::memcpy(result->data(), m.data(), result->size() * sizeof(float_type));
    return result;
}

} // namespace internal

using float_type = internal::float_type;
using float_vec = internal::float_vec;
using shared_float_vec = internal::shared_float_vec;
using tensor5 = internal::tensor5;
using tensor5s = internal::tensor5s;
using tensor5s_vec = internal::tensor5s_vec;


inline std::string show_tensor5(const tensor5& t)
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

inline std::string show_tensor5s(const tensor5s& ts)
{
    return fplus::show_cont(fplus::transform(show_tensor5, ts));
}

// Converts a memory block holding 8-bit values into a tensor5.
// Data must be stored row-wise (and channels_last).
// Scales the values from range [0, 255] into [low, high].
// May be used to convert an image (bgr, rgba, gray, etc.) to a tensor5.
inline tensor5 tensor5_from_bytes(const std::uint8_t* value_ptr,
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
    return tensor5(shape5(1, 1, height, width, channels), std::move(values));
}

// Converts a tensor5 into a memory block holding 8-bit values.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
// May be used to convert a tensor5 into an image.
inline void tensor5_into_bytes(const tensor5& t, std::uint8_t* value_ptr,
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

// Converts a tensor5 into a vector of bytes.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
inline std::vector<std::uint8_t> tensor5_to_bytes(const tensor5& t,
    internal::float_type low = 0.0f, internal::float_type high = 1.0f)
{
    std::vector<std::uint8_t> bytes(t.shape().volume(), 0);
    tensor5_into_bytes(t, bytes.data(), bytes.size(), low, high);
    return bytes;
}

inline tensor5s_vec reshape_tensor5_vectors(
    std::size_t vectors_size,
    std::size_t vector_size,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const tensor5s_vec& tss)
{
    const auto values = fplus::concat(fplus::concat(
        fplus::transform_inner(
            [](const tensor5& t) -> float_vec {return *t.as_vector();},
            tss)));

    fdeep::internal::assertion(values.size() == vectors_size * vector_size * height * width * depth,
        "Invalid number of values for reshape target.");

    const auto ts = fplus::transform(
        [&](fdeep::float_vec v) -> tensor5 {return tensor5(shape5(1, 1, height, width, depth), std::move(v));},
        fplus::split_every(depth * height * width, values));

    return fplus::split_every(vector_size, ts);
}

} // namespace fdeep
