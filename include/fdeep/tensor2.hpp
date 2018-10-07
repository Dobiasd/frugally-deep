// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape_hw.hpp"
#include "fdeep/tensor2_pos_yx.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace fdeep { namespace internal
{

class tensor2
{
public:
    tensor2(const shape_hw& shape, const shared_float_vec& values) :
        shape_(shape),
        values_(values)
    {
        assertion(shape.area() == values->size(), "invalid number of values");
    }
    tensor2(const shape_hw& shape, float_vec&& values) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(std::move(values)))
    {
        assertion(shape.area() == values_->size(), "invalid number of values");
    }
    tensor2(const shape_hw& shape, float_type value) :
        shape_(shape),
        values_(fplus::make_shared_ref<float_vec>(shape.area(), value))
    {
    }

    const float_type& get(const tensor2_pos_yx& pos) const
    {
        return (*values_)[idx(pos)];
    }
    const float_type& get_yx(std::size_t y, std::size_t x) const
    {
        return get(tensor2_pos_yx(y, x));
    }
    void set(const tensor2_pos_yx& pos, float_type value)
    {
        (*values_)[idx(pos)] = value;
    }
    void set_yx(std::size_t y, std::size_t x, float_type value)
    {
        set(tensor2_pos_yx(y, x), value);
    }
    const shape_hw& shape() const
    {
        return shape_;
    }
    const shared_float_vec& as_vector() const
    {
        return values_;
    }

private:
    std::size_t idx(const tensor2_pos_yx& pos) const
    {
        return
            pos.y_ * shape().width_ +
            pos.x_;
    };
    shape_hw shape_;
    shared_float_vec values_;
};

template <typename F>
tensor2 transform_tensor2(F f, const tensor2& m)
{
    return tensor2(m.shape(), fplus::transform(f, m.as_vector()));
}

inline tensor2 add_to_tensor2_elems(const tensor2& m, float_type x)
{
    return tensor2(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return x + e;
    }, *m.as_vector()));
}

inline tensor2 sub_from_tensor2_elems(const tensor2& m, float_type x)
{
    return tensor2(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return e - x;
    }, *m.as_vector()));
}

inline tensor2 multiply_tensor2_elems(const tensor2& m, float_type x)
{
    return tensor2(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return x * e;
    }, *m.as_vector()));
}

inline tensor2 divide_tensor2_elems(const tensor2& m, float_type x)
{
    return tensor2(m.shape(), fplus::transform([x](float_type e) -> float_type
    {
        return e / x;
    }, *m.as_vector()));
}

inline shared_float_vec eigen_row_major_mat_to_values(const RowMajorMatrixXf& m)
{
    shared_float_vec result = fplus::make_shared_ref<float_vec>();
    result->resize(static_cast<std::size_t>(m.rows() * m.cols()));
    std::memcpy(result->data(), m.data(), result->size() * sizeof(float_type));
    return result;
}

inline RowMajorMatrixXf eigen_row_major_mat_from_values(std::size_t height,
    std::size_t width, const float_vec& values)
{
    assertion(height * width == values.size(), "invalid shape");
    RowMajorMatrixXf m(height, width);
    std::memcpy(m.data(), values.data(), values.size() * sizeof(float_type));
    return m;
}

} } // namespace fdeep, namespace internal
