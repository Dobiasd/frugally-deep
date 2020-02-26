// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/tensor_pos.hpp"
#include "fdeep/tensor_shape.hpp"

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

class tensor
{
public:
    tensor(const tensor_shape& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.volume() == values->size(),
            std::string("invalid number of values. shape: ") +
            show_tensor_shape(shape) + "; value count: " +
            std::to_string(values->size()));
    }
    tensor(const tensor_shape& shape, float_vec&& values) :
        tensor(shape, fplus::make_shared_ref<float_vec>(std::move(values)))
    {
    }
    tensor(const tensor_shape& shape, float_type value) :
        tensor(shape, fplus::make_shared_ref<float_vec>(shape.volume(), value))
    {
    }
    float_type get(const tensor_pos& pos) const
    {
        return (*values_)[idx(pos)];
    }
    float_type get_ignore_rank(const tensor_pos& pos) const
    {
        return (*values_)[idx_ignore_rank(pos)];
    }
    float_type get_y_x_padded(float_type pad_value,
        int y, int x, std::size_t z) const
    {
        if (y < 0 || y >= static_cast<int>(shape().height_) ||
            x < 0 || x >= static_cast<int>(shape().width_))
        {
            return pad_value;
        }
        return get_ignore_rank(tensor_pos(static_cast<std::size_t>(y), static_cast<std::size_t>(x), z));
    }
    float_type get_x_z_padded(float_type pad_value,
        std::size_t y, int x, int z) const
    {
        if (x < 0 || x >= static_cast<int>(shape().width_) ||
            z < 0 || z >= static_cast<int>(shape().depth_))
        {
            return pad_value;
        }
        return get(tensor_pos(y, static_cast<std::size_t>(x), static_cast<std::size_t>(z)));
    }
    void set(const tensor_pos& pos, float_type value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set_ignore_rank(const tensor_pos& pos, float_type value)
    {
        (*values_)[idx_ignore_rank(pos)] = value;
    }

    // Deprecated! Will likely be removed from the API soon.
    // Please use
    //     get(const tensor_pos&) const
    // or
    //     get_ignore_rank(const tensor_pos&) const
    // instead.
    float_type get(std::size_t pos_dim_5, std::size_t pos_dim_4,
        std::size_t y, std::size_t x, std::size_t z) const
    {
        return get_ignore_rank(tensor_pos(pos_dim_5, pos_dim_4, y, x, z));
    }

    // Deprecated! Will likely be removed from the API soon.
    // Please use
    //     set(const tensor_pos, float_type)
    // or
    //     set_ignore_rank(const tensor_pos&, float_type)
    // instead.
    void set(std::size_t pos_dim_5, std::size_t pos_dim_4,
        std::size_t y, std::size_t x, std::size_t z, float_type value)
    {
        set_ignore_rank(tensor_pos(pos_dim_5, pos_dim_4, y, x, z), value);
    }

    const tensor_shape& shape() const
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
    std::size_t idx_ignore_rank(const tensor_pos& pos) const
    {
        return
            pos.pos_dim_5_ * shape().size_dim_4_ * shape().height_ * shape().width_ * shape().depth_ +
            pos.pos_dim_4_ * shape().height_ * shape().width_ * shape().depth_ +
            pos.y_ * shape().width_ * shape().depth_ +
            pos.x_ * shape().depth_ +
            pos.z_;
    };
    std::size_t idx(const tensor_pos& pos) const
    {
        assertion(pos.rank() == shape().rank(), "Invalid position rank for tensor");
        return idx_ignore_rank(pos);
    };
    tensor_shape shape_;
    shared_float_vec values_;
};

typedef std::vector<tensor> tensors;
typedef std::vector<tensors> tensors_vec;

inline tensor single_tensor_from_tensors(const tensors& ts)
{
    assertion(ts.size() == 1, "invalid number of tensors");
    return ts.front();
}

inline bool is_singleton_value(const tensor& t)
{
    return t.shape().volume() == 1;
}

inline float_type to_singleton_value(const tensor& t)
{
    assertion(is_singleton_value(t), "Tensor must contain exactly one value.");
    return t.get(tensor_pos(static_cast<std::size_t>(0)));
}

inline tensor from_singleton_value(float_type value)
{
    return tensor(tensor_shape(static_cast<std::size_t>(1)), value);
}

inline tensor tensor_with_changed_rank(const tensor& t, std::size_t rank)
{
    return tensor(tensor_shape_with_changed_rank(t.shape(), rank), t.as_vector());
}

template <typename F>
tensor transform_tensor(F f, const tensor& m)
{
    return tensor(m.shape(), fplus::transform(f, *m.as_vector()));
}

inline tensor tensor_from_depth_slices(const std::vector<tensor>& ms)
{
    assertion(!ms.empty(), "no slices given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor, shape, tensor_shape), ms),
        "all slices must have the same size");
    for (const auto& m : ms)
    {
        m.shape().assert_is_shape_2();
    }
    std::size_t height = ms.front().shape().height_;
    std::size_t width = ms.front().shape().width_;
    tensor m(tensor_shape(height, width, ms.size()), 0);
    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < m.shape().depth_; ++z)
            {
                m.set(tensor_pos(y, x, z),
                    ms[z].get(tensor_pos(y, x, 0)));
            }
        }
    }
    return m;
}

inline std::vector<tensor> tensor_to_depth_slices(const tensor& m)
{
    std::vector<tensor> ms;
    ms.reserve(m.shape().depth_);
    for (std::size_t i = 0; i < m.shape().depth_; ++i)
    {
        ms.push_back(tensor(change_tensor_shape_dimension_by_index(
            m.shape(), 4, 1),
            0));
    }

    for (std::size_t y = 0; y < m.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < m.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < m.shape().depth_; ++z)
            {
                // .set and .get would work here too
                // but using _ignore_rank here for
                // improved performance of depthwise_conv_2d_layer
                ms[z].set_ignore_rank(tensor_pos(y, x, 0),
                    m.get_ignore_rank(tensor_pos(y, x, z)));
            }
        }
    }
    return ms;
}

inline tensors tensor_to_tensors_width_slices(const tensor& m)
{
    tensors ms;
    ms.reserve(m.shape().width_);
    for (std::size_t i = 0; i < m.shape().width_; ++i)
    {
        ms.push_back(tensor(change_tensor_shape_dimension_by_index(
            m.shape(), 3, 1),
            0));
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
                        ms[x].set_ignore_rank(tensor_pos(dim5, dim4, y, 0, z),
                            m.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensors tensor_to_tensors_height_slices(const tensor& m)
{
    tensors ms;
    ms.reserve(m.shape().height_);
    for (std::size_t i = 0; i < m.shape().height_; ++i)
    {
        ms.push_back(tensor(change_tensor_shape_dimension_by_index(
            m.shape(), 2, 1),
            0));
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
                        ms[y].set_ignore_rank(tensor_pos(dim5, dim4, 0, x, z),
                            m.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensors tensor_to_tensors_dim4_slices(const tensor& m)
{
    tensors ms;
    ms.reserve(m.shape().size_dim_4_);
    for (std::size_t i = 0; i < m.shape().size_dim_4_; ++i)
    {
        ms.push_back(tensor(change_tensor_shape_dimension_by_index(
            m.shape(), 1, 1),
            0));
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
                        ms[dim4].set_ignore_rank(tensor_pos(dim5, 0, y, x, z),
                            m.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline tensors tensor_to_tensors_dim5_slices(const tensor& m)
{
    tensors ms;
    ms.reserve(m.shape().size_dim_5_);
    for (std::size_t i = 0; i < m.shape().size_dim_5_; ++i)
    {
        ms.push_back(tensor(change_tensor_shape_dimension_by_index(
            m.shape(), 0, 1),
            0));
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
                        ms[dim5].set_ignore_rank(tensor_pos(dim4, y, x, z),
                            m.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                    }
                }
            }
        }
    }
    return ms;
}

inline std::pair<tensor_pos, tensor_pos> tensor_min_max_pos(
    const tensor& vol)
{
    tensor_pos result_min(0, 0, 0, 0, 0);
    tensor_pos result_max(0, 0, 0, 0, 0);
    float_type value_max = std::numeric_limits<float_type>::lowest();
    float_type value_min = std::numeric_limits<float_type>::max();
    for (std::size_t dim5 = 0; dim5 < vol.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < vol.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < vol.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < vol.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < vol.shape().depth_; ++z)
                    {
                        auto current_value = vol.get_ignore_rank(tensor_pos(y, x, z));
                        if (current_value > value_max)
                        {
                            result_max = tensor_pos(dim5, dim4, y, x, z);
                            value_max = current_value;
                        }
                        if (current_value < value_min)
                        {
                            result_min = tensor_pos(dim5, dim4, y, x, z);
                            value_min = current_value;
                        }
                    }
                }
            }
        }
    }
    return std::make_pair(
        tensor_pos_with_changed_rank(result_min, vol.shape().rank()),
        tensor_pos_with_changed_rank(result_max, vol.shape().rank()));
}

inline std::vector<std::vector<std::size_t>> get_tensors_shape_sizes(const tensors& ts)
{
    return {
        fplus::transform([](const auto& t) { return t.shape().size_dim_5_; }, ts),
        fplus::transform([](const auto& t) { return t.shape().size_dim_4_; }, ts),
        fplus::transform([](const auto& t) { return t.shape().height_; }, ts),
        fplus::transform([](const auto& t) { return t.shape().width_; }, ts),
        fplus::transform([](const auto& t) { return t.shape().depth_; }, ts)
    };
}

inline tensor_pos tensor_max_pos(const tensor& vol)
{
    return tensor_min_max_pos(vol).second;
}

inline tensor concatenate_tensors_depth(const tensors& in)
{
    const auto shape_sizes = get_tensors_shape_sizes(in);
    assertion(
        fplus::all_the_same(shape_sizes[0]) &&
        fplus::all_the_same(shape_sizes[1]) &&
        fplus::all_the_same(shape_sizes[2]) &&
        fplus::all_the_same(shape_sizes[3]),
        "Tensor shapes differ on wrong dimension.");

    tensor result(change_tensor_shape_dimension_by_index(
            in.front().shape(), 4, fplus::sum(shape_sizes[4])),
        0);
    std::size_t out_dim1 = 0;
    for (const auto& t: in)
    {
        for (std::size_t z = 0; z < t.shape().depth_; ++z, ++out_dim1)
        {
            for (std::size_t dim5 = 0; dim5 < t.shape().size_dim_5_; ++dim5)
            {
                for (std::size_t dim4 = 0; dim4 < t.shape().size_dim_4_; ++dim4)
                {
                    for (std::size_t y = 0; y < t.shape().height_; ++y)
                    {
                        for (std::size_t x = 0; x < t.shape().width_; ++x)
                        {
                            result.set_ignore_rank(tensor_pos(dim5, dim4, y, x, out_dim1),
                            t.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline tensor concatenate_tensors_width(const tensors& in)
{
    const auto shape_sizes = get_tensors_shape_sizes(in);
    assertion(
        fplus::all_the_same(shape_sizes[0]) &&
        fplus::all_the_same(shape_sizes[1]) &&
        fplus::all_the_same(shape_sizes[2]) &&
        fplus::all_the_same(shape_sizes[4]),
        "Tensor shapes differ on wrong dimension.");

    tensor result(change_tensor_shape_dimension_by_index(
            in.front().shape(), 3, fplus::sum(shape_sizes[3])),
        0);
    std::size_t out_dim2 = 0;
    for (const auto& t: in)
    {
        for (std::size_t x = 0; x < t.shape().width_; ++x, ++out_dim2)
        {
            for (std::size_t dim5 = 0; dim5 < t.shape().size_dim_5_; ++dim5)
            {
                for (std::size_t dim4 = 0; dim4 < t.shape().size_dim_4_; ++dim4)
                {
                    for (std::size_t y = 0; y < t.shape().height_; ++y)
                    {
                        for (std::size_t z = 0; z < t.shape().depth_; ++z)
                        {
                            result.set_ignore_rank(tensor_pos(dim5, dim4, y, out_dim2, z),
                                t.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline tensor concatenate_tensors_height(const tensors& in)
{
    const auto shape_sizes = get_tensors_shape_sizes(in);
    assertion(
        fplus::all_the_same(shape_sizes[0]) &&
        fplus::all_the_same(shape_sizes[1]) &&
        fplus::all_the_same(shape_sizes[3]) &&
        fplus::all_the_same(shape_sizes[4]),
        "Tensor shapes differ on wrong dimension.");

    tensor result(change_tensor_shape_dimension_by_index(
            in.front().shape(), 2, fplus::sum(shape_sizes[2])),
        0);
    std::size_t out_dim3 = 0;
    for (const auto& t: in)
    {
        for (std::size_t y = 0; y < t.shape().height_; ++y, ++out_dim3)
        {
            for (std::size_t dim5 = 0; dim5 < t.shape().size_dim_5_; ++dim5)
            {
                for (std::size_t dim4 = 0; dim4 < t.shape().size_dim_4_; ++dim4)
                {
                    for (std::size_t x = 0; x < t.shape().width_; ++x)
                    {
                        for (std::size_t z = 0; z < t.shape().depth_; ++z)
                        {
                            result.set_ignore_rank(tensor_pos(dim5, dim4, out_dim3, x, z),
                                t.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline tensor concatenate_tensors_dim4(const tensors& in)
{
    const auto shape_sizes = get_tensors_shape_sizes(in);
    assertion(
        fplus::all_the_same(shape_sizes[0]) &&
        fplus::all_the_same(shape_sizes[2]) &&
        fplus::all_the_same(shape_sizes[3]) &&
        fplus::all_the_same(shape_sizes[4]),
        "Tensor shapes differ on wrong dimension.");
    tensor result(change_tensor_shape_dimension_by_index(
            in.front().shape(), 1, fplus::sum(shape_sizes[1])),
        0);
    std::size_t out_dim4 = 0;
    for (const auto& t: in)
    {
        for (std::size_t dim4 = 0; dim4 < t.shape().size_dim_4_; ++dim4, ++out_dim4)
        {
            for (std::size_t dim5 = 0; dim5 < t.shape().size_dim_5_; ++dim5)
            {
                for (std::size_t y = 0; y < t.shape().height_; ++y)
                {
                    for (std::size_t x = 0; x < t.shape().width_; ++x)
                    {
                        for (std::size_t z = 0; z < t.shape().depth_; ++z)
                        {
                            result.set_ignore_rank(tensor_pos(dim5, out_dim4, y, x, z),
                                t.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline tensor concatenate_tensors_dim5(const tensors& in)
{
    const auto shape_sizes = get_tensors_shape_sizes(in);
    assertion(
        fplus::all_the_same(shape_sizes[1]) &&
        fplus::all_the_same(shape_sizes[2]) &&
        fplus::all_the_same(shape_sizes[3]) &&
        fplus::all_the_same(shape_sizes[4]),
        "Tensor shapes differ on wrong dimension.");

    tensor result(change_tensor_shape_dimension_by_index(
            in.front().shape(), 0, fplus::sum(shape_sizes[0])),
        0);
    std::size_t out_dim5 = 0;
    for (const auto& t: in)
    {
        for (std::size_t dim5 = 0; dim5 < t.shape().size_dim_5_; ++dim5, ++out_dim5)
        {
            for (std::size_t dim4 = 0; dim4 < t.shape().size_dim_4_; ++dim4)
            {
                for (std::size_t y = 0; y < t.shape().height_; ++y)
                {
                    for (std::size_t x = 0; x < t.shape().width_; ++x)
                    {
                        for (std::size_t z = 0; z < t.shape().depth_; ++z)
                        {
                            result.set_ignore_rank(tensor_pos(out_dim5, dim4, y, x, z),
                                t.get_ignore_rank(tensor_pos(dim5, dim4, y, x, z)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

inline tensor concatenate_tensors(const tensors& ts, std::int32_t axis)
{
    const auto rank = ts.front().shape().rank();
    if (axis < 0)
    {
        axis = axis + static_cast<std::int32_t>(rank) + 1;
    }
    axis = std::min(5, axis - static_cast<std::int32_t>(rank) + 5);
    if (axis == 5)
    {
        return concatenate_tensors_depth(ts);
    }
    if (axis == 4)
    {
        return concatenate_tensors_width(ts);
    }
    if (axis == 3)
    {
        return concatenate_tensors_height(ts);
    }
    if (axis == 2)
    {
        return concatenate_tensors_dim4(ts);
    }
    if (axis == 1)
    {
        return concatenate_tensors_dim5(ts);
    }
    raise_error("Invalid axis (" + std::to_string(axis) +
        ") for tensor concatenation.");
    return tensor(tensor_shape(static_cast<std::size_t>(0)), 0);
}

inline tensor flatten_tensor(const tensor& vol)
{
    return tensor(tensor_shape(vol.shape().volume()), vol.as_vector());
}

inline tensor pad_tensor(float_type val,
    std::size_t top_pad, std::size_t bottom_pad,
    std::size_t left_pad, std::size_t right_pad,
    const tensor& in)
{
    tensor result(tensor_shape_with_changed_rank(tensor_shape(
        in.shape().height_ + top_pad + bottom_pad,
        in.shape().width_ + left_pad + right_pad,
        in.shape().depth_), in.shape().rank()), val);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set_ignore_rank(tensor_pos(y + top_pad, x + left_pad, z),
                    in.get_ignore_rank(tensor_pos(y, x, z)));
            }
        }
    }
    return result;
}

inline void check_permute_tensor_dims(const std::vector<std::size_t>& dims_raw)
{
    assertion(
        fplus::minimum(dims_raw) >= 1 &&
        fplus::maximum(dims_raw) <= 5 &&
        fplus::size_of_cont(fplus::nub(dims_raw)) ==
            fplus::size_of_cont(dims_raw),
            "Invalid dims for permute_tensor.");
}

inline tensor permute_tensor(const tensor& in,
    const std::vector<std::size_t>& dims_raw)
{
    check_permute_tensor_dims(dims_raw);

    const auto dims = fplus::transform(fplus::subtract<std::size_t>(1), dims_raw);

    const auto permute_idxs = [&dims](const std::vector<std::size_t>& idxs) {
        return fplus::elems_at_idxs(dims, idxs);
    };
    const auto out_shape = create_tensor_shape_from_dims(
        permute_idxs(in.shape().dimensions()));

    tensor out(out_shape, 0);

    for (std::size_t dim5 = 0; dim5 < in.shape().size_dim_5_; ++dim5)
    {
        for (std::size_t dim4 = 0; dim4 < in.shape().size_dim_4_; ++dim4)
        {
            for (std::size_t y = 0; y < in.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < in.shape().depth_; ++z)
                    {
                        const auto in_pos = tensor_pos_with_changed_rank(
                            tensor_pos(dim5, dim4, y, x, z), dims.size());
                        const auto out_pos = create_tensor_pos_from_dims(
                            permute_idxs(in_pos.dimensions()));
                        out.set(out_pos, in.get(in_pos));
                    }
                }
            }
        }
    }
    return out;
}

inline tensor crop_tensor(
    std::size_t top_crop, std::size_t bottom_crop,
    std::size_t left_crop, std::size_t right_crop,
    const tensor& in)
{
    tensor result(tensor_shape_with_changed_rank(tensor_shape(
        in.shape().height_ - (top_crop + bottom_crop),
        in.shape().width_ - (left_crop + right_crop),
        in.shape().depth_), in.shape().rank()), 0);
    for (std::size_t y = 0; y < result.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < result.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < result.shape().depth_; ++z)
            {
                result.set_ignore_rank(tensor_pos(y, x, z),
                    in.get_ignore_rank(tensor_pos(y + top_crop, x + left_crop, z)));
            }
        }
    }
    return result;
}

inline tensor dilate_tensor(const shape2& dilation_rate, const tensor& in)
{
    assertion(in.shape().rank() <= 3, "Invalid rank for dilation");
    if (dilation_rate == shape2(1, 1))
    {
        return in;
    }

    tensor result(dilate_tensor_shape(dilation_rate, in.shape()), 0);
    for (std::size_t y = 0; y < in.shape().height_; ++y)
    {
        for (std::size_t x = 0; x < in.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < in.shape().depth_; ++z)
            {
                result.set(tensor_pos(
                    y * dilation_rate.height_,
                    x * dilation_rate.width_,
                    z),
                    in.get(tensor_pos(y, x, z)));
            }
        }
    }
    return result;
}

inline tensor sum_tensors(const tensors& ts)
{
    assertion(!ts.empty(), "no tensors given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor, shape, tensor_shape), ts),
        "all tensors must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor, as_vector, shared_float_vec), ts);
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
    return tensor(ts.front().shape(), std::move(result_values));
}

inline tensor multiply_tensors(const tensors& ts_all)
{
    assertion(!ts_all.empty(), "no tensors given");
    const auto factors_and_tensors = fplus::partition(
        is_singleton_value, ts_all);
    auto ts = factors_and_tensors.second;
    if (ts.empty()) {
        ts.push_back(from_singleton_value(static_cast<float_type>(1)));
    }
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor, shape, tensor_shape), ts),
        "all tensors must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor, as_vector, shared_float_vec), ts);
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
    if (factors_and_tensors.first.size() > 0) {
        const auto factor = fplus::product(
            fplus::transform(to_singleton_value, factors_and_tensors.first));
        result_values = fplus::transform(
            fplus::multiply_with<float_type>(factor), result_values);
    }
    return tensor(ts.front().shape(), std::move(result_values));
}

inline tensor subtract_tensor(const tensor& a, const tensor& b)
{
    assertion(a.shape() == b.shape(),
        "both tensors must have the same size");
    auto result_values = fplus::zip_with(std::minus<float_type>(),
        *a.as_vector(), *b.as_vector());
    return tensor(a.shape(), std::move(result_values));
}

inline tensor average_tensors(const tensors& ts)
{
    const auto sum = sum_tensors(ts);
    const float_type divisor = static_cast<float_type>(ts.size());
    return transform_tensor(fplus::multiply_with(1 / divisor), sum);
}

inline tensor max_tensors(const tensors& ts)
{
    assertion(!ts.empty(), "no tensors given");
    assertion(
        fplus::all_the_same_on(fplus_c_mem_fn_t(tensor, shape, tensor_shape), ts),
        "all tensors must have the same size");
    const auto ts_values = fplus::transform(
        fplus_c_mem_fn_t(tensor, as_vector, shared_float_vec), ts);
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
    return tensor(ts.front().shape(), std::move(result_values));
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
using tensor = internal::tensor;
using tensors = internal::tensors;
using tensors_vec = internal::tensors_vec;


inline std::string show_tensor(const tensor& t)
{
    const auto xs = *t.as_vector();
    const auto test_strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', 0, 4), xs);
    const auto max_length = fplus::size_of_cont(fplus::maximum_on(
        fplus::size_of_cont<std::string>, test_strs));
    const auto strs = fplus::transform(
        fplus::fwd::show_float_fill_left(' ', max_length, 4), xs);
    return fplus::show_cont(
        fplus::split_every(t.shape().size_dim_5_,
            fplus::split_every(t.shape().size_dim_4_,
                fplus::split_every(t.shape().height_,
                    fplus::split_every(t.shape().width_,
                        fplus::split_every(t.shape().depth_, strs))))));
}

inline std::string show_tensors(const tensors& ts)
{
    return fplus::show_cont(fplus::transform(show_tensor, ts));
}

// Converts a memory block holding 8-bit values into a tensor.
// Data must be stored row-wise (and channels_last).
// Scales the values from range [0, 255] into [low, high].
// Example:
//     With low = 0.0 and high = 1.0 every value is essentially divided by 255.
// May be used to convert an image (bgr, rgba, gray, etc.) to a tensor.
inline tensor tensor_from_bytes(const std::uint8_t* value_ptr,
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
    return tensor(tensor_shape(height, width, channels), std::move(values));
}

// Converts a tensor into a memory block holding 8-bit values.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
// May be used to convert a tensor into an image.
inline void tensor_into_bytes(const tensor& t, std::uint8_t* value_ptr,
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

// Converts a tensor into a vector of bytes.
// Data will be stored row-wise (and channels_last).
// Scales the values from range [low, high] into [0, 255].
inline std::vector<std::uint8_t> tensor_to_bytes(const tensor& t,
    internal::float_type low = 0.0f, internal::float_type high = 1.0f)
{
    std::vector<std::uint8_t> bytes(t.shape().volume(), 0);
    tensor_into_bytes(t, bytes.data(), bytes.size(), low, high);
    return bytes;
}

inline tensors_vec reshape_tensor_vectors(
    std::size_t vectors_size,
    std::size_t vector_size,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const tensors_vec& tss)
{
    const auto values = fplus::concat(fplus::concat(
        fplus::transform_inner(
            [](const tensor& t) -> float_vec {return *t.as_vector();},
            tss)));

    fdeep::internal::assertion(values.size() == vectors_size * vector_size * height * width * depth,
        "Invalid number of values for reshape target.");

    const auto ts = fplus::transform(
        [&](fdeep::float_vec v) -> tensor {return tensor(tensor_shape(height, width, depth), std::move(v));},
        fplus::split_every(depth * height * width, values));

    return fplus::split_every(vector_size, ts);
}

} // namespace fdeep
