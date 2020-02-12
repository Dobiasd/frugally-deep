// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2.hpp"
#include "fdeep/tensor_shape_variable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class tensor_shape
{
public:
    // The outer (left-most) dimensions are not used for batch prediction.
    // If you like to do multiple forward passes on a model at once,
    // use fdeep::model::predict_multi instead.
    explicit tensor_shape(
        std::size_t size_dim_5,
        std::size_t size_dim_4,
        std::size_t height,
        std::size_t width,
        std::size_t depth) :
            rank_(5),
            size_dim_5_(size_dim_5),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape(
        std::size_t size_dim_4,
        std::size_t height,
        std::size_t width,
        std::size_t depth) :
            rank_(4),
            size_dim_5_(1),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape(
        std::size_t height,
        std::size_t width,
        std::size_t depth) :
            rank_(3),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape(
        std::size_t width,
        std::size_t depth) :
            rank_(2),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(1),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape(
        std::size_t depth) :
            rank_(1),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(1),
            width_(1),
            depth_(depth)
    {
    }

    std::size_t volume() const
    {
        return size_dim_5_ * size_dim_4_ * height_ * width_ * depth_;
    }

    void assert_is_shape_2() const
    {
        assertion(
            size_dim_5_ == 1 && size_dim_4_ == 1 && depth_ == 1,
            "Only height and width may be not equal 1.");
    }

    void assert_is_shape_3() const
    {
        assertion(
            size_dim_5_ == 1 && size_dim_4_ == 1,
            "Only height, width and depth may be not equal 1.");
    }

    shape2 without_depth() const
    {
        assert_is_shape_3();
        return shape2(height_, width_);
    }

    std::size_t rank_;
    std::size_t size_dim_5_;
    std::size_t size_dim_4_;
    std::size_t height_;
    std::size_t width_;
    std::size_t depth_;
};

inline tensor_shape make_tensor_shape_with(
    const tensor_shape& default_shape,
    const tensor_shape_variable shape)
{
    if (shape.rank_ == 1)
        return tensor_shape(
            fplus::just_with_default(default_shape.depth_, shape.depth_));
    if (shape.rank_ == 2)
        return tensor_shape(
            fplus::just_with_default(default_shape.width_, shape.width_),
            fplus::just_with_default(default_shape.depth_, shape.depth_));
    if (shape.rank_ == 3)
        return tensor_shape(
            fplus::just_with_default(default_shape.height_, shape.height_),
            fplus::just_with_default(default_shape.width_, shape.width_),
            fplus::just_with_default(default_shape.depth_, shape.depth_));
    if (shape.rank_ == 4)
        return tensor_shape(
            fplus::just_with_default(default_shape.size_dim_4_, shape.size_dim_4_),
            fplus::just_with_default(default_shape.height_, shape.height_),
            fplus::just_with_default(default_shape.width_, shape.width_),
            fplus::just_with_default(default_shape.depth_, shape.depth_));
    else
        return tensor_shape(
            fplus::just_with_default(default_shape.size_dim_5_, shape.size_dim_5_),
            fplus::just_with_default(default_shape.size_dim_4_, shape.size_dim_4_),
            fplus::just_with_default(default_shape.height_, shape.height_),
            fplus::just_with_default(default_shape.width_, shape.width_),
            fplus::just_with_default(default_shape.depth_, shape.depth_));
}

inline bool tensor_shape_equals_tensor_shape_variable(
    const tensor_shape& lhs, const tensor_shape_variable& rhs)
{
    return
        (rhs.rank_ == lhs.rank_) &&
        (rhs.size_dim_5_.is_nothing() || lhs.size_dim_5_ == rhs.size_dim_5_.unsafe_get_just()) &&
        (rhs.size_dim_4_.is_nothing() || lhs.size_dim_4_ == rhs.size_dim_4_.unsafe_get_just()) &&
        (rhs.height_.is_nothing() || lhs.height_ == rhs.height_.unsafe_get_just()) &&
        (rhs.width_.is_nothing() || lhs.width_ == rhs.width_.unsafe_get_just()) &&
        (rhs.depth_.is_nothing() || lhs.depth_ == rhs.depth_.unsafe_get_just());
}

inline bool operator == (const tensor_shape& lhs, const tensor_shape_variable& rhs)
{
    return tensor_shape_equals_tensor_shape_variable(lhs, rhs);
}

inline bool operator == (const std::vector<tensor_shape>& lhss,
    const std::vector<tensor_shape_variable>& rhss)
{
    return fplus::all(fplus::zip_with(tensor_shape_equals_tensor_shape_variable,
        lhss, rhss));
}

inline bool operator == (const tensor_shape& lhs, const tensor_shape& rhs)
{
    return
        lhs.rank_ == rhs.rank_ &&
        lhs.size_dim_5_ == rhs.size_dim_5_ &&
        lhs.size_dim_4_ == rhs.size_dim_4_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline tensor_shape tensor_shape_with_changed_rank(const tensor_shape& s, std::size_t rank)
{
    assertion(rank >= 1 && rank <= 5, "Invalid target rank");
    if (rank == 4)
    {
        assertion(s.size_dim_5_ == 1, "Invalid target rank");
        return tensor_shape(s.size_dim_4_, s.height_, s.width_, s.depth_);
    }
    if (rank == 3)
    {
        assertion(s.size_dim_5_ == 1, "Invalid target rank");
        assertion(s.size_dim_4_ == 1, "Invalid target rank");
        return tensor_shape(s.height_, s.width_, s.depth_);
    }
    if (rank == 2)
    {
        assertion(s.size_dim_5_ == 1, "Invalid target rank");
        assertion(s.size_dim_4_ == 1, "Invalid target rank");
        assertion(s.height_ == 1, "Invalid target rank");
        return tensor_shape(s.width_, s.depth_);
    }
    if (rank == 1)
    {
        assertion(s.size_dim_5_ == 1, "Invalid target rank");
        assertion(s.size_dim_4_ == 1, "Invalid target rank");
        assertion(s.height_ == 1, "Invalid target rank");
        assertion(s.width_ == 1, "Invalid target rank");
        return tensor_shape(s.depth_);
    }
    return tensor_shape(s.size_dim_5_, s.size_dim_4_, s.height_, s.width_, s.depth_);
}

inline tensor_shape dilate_tensor_shape(
    const shape2& dilation_rate, const tensor_shape& s)
{
    assertion(dilation_rate.height_ >= 1, "invalid dilation rate");
    assertion(dilation_rate.width_ >= 1, "invalid dilation rate");

    const std::size_t height = s.height_ +
        (s.height_ - 1) * (dilation_rate.height_ - 1);
    const std::size_t width = s.width_ +
        (s.width_ - 1) * (dilation_rate.width_ - 1);
    return tensor_shape_with_changed_rank(
        tensor_shape(s.size_dim_5_, s.size_dim_4_, height, width, s.depth_),
        s.rank_
    );
}

inline std::size_t get_tensor_shape_dimension_by_index(const tensor_shape& s,
    const std::size_t idx)
{
    if (idx == 0)
        return s.size_dim_5_;
    if (idx == 1)
        return s.size_dim_4_;
    if (idx == 2)
        return s.height_;
    if (idx == 3)
        return s.width_;
    if (idx == 4)
        return s.depth_;
    raise_error("Invalid tensor_shape index.");
    return 0;
}

inline tensor_shape change_tensor_shape_dimension_by_index(const tensor_shape& in,
    const std::size_t idx, const std::size_t dim)
{
    tensor_shape out = in;
    if (idx == 0)
    {
        out.size_dim_5_ = dim;
        out.rank_ = std::max<std::size_t>(in.rank_, 5);
    }
    else if (idx == 1)
    {
        out.size_dim_4_ = dim;
        out.rank_ = std::max<std::size_t>(in.rank_, 4);
    }
    else if (idx == 2)
    {
        out.height_ = dim;
        out.rank_ = std::max<std::size_t>(in.rank_, 3);
    }
    else if (idx == 3)
    {
        out.width_ = dim;
        out.rank_ = std::max<std::size_t>(in.rank_, 2);
    }
    else if (idx == 4)
    {
        out.depth_ = dim;
        out.rank_ = std::max<std::size_t>(in.rank_, 1);
    }
    else
        raise_error("Invalid tensor_shape index.");
    return out;
}

} // namespace internal

using tensor_shape = internal::tensor_shape;

inline std::string show_tensor_shape(const tensor_shape& s)
{
    const std::vector<std::size_t> dimensions = {
        s.size_dim_5_,
        s.size_dim_4_,
        s.height_,
        s.width_,
        s.depth_
        };
    return std::to_string(s.rank_) + fplus::show_cont_with_frame(", ", "(", ")", dimensions);
}

inline std::string show_tensor_shapes(
    const std::vector<tensor_shape>& shapes)
{
    return fplus::show_cont(fplus::transform(show_tensor_shape, shapes));
}

} // namespace fdeep
