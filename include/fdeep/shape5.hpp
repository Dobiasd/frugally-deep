// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2.hpp"
#include "fdeep/shape5_variable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class shape5
{
public:
    explicit shape5(
        std::size_t size_dim_5,
        std::size_t size_dim_4,
        std::size_t height,
        std::size_t width,
        std::size_t depth) :
            size_dim_5_(size_dim_5),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

    std::size_t volume() const
    {
        return size_dim_5_ * size_dim_4_ * height_ * width_ * depth_;
    }

    shape2 without_depth() const
    {
        assertion(
            size_dim_5_ == 1 && size_dim_4_ == 1,
            "Only last three dimensions may be non-zero.");
        return shape2(height_, width_);
    }

    std::size_t size_dim_5_;
    std::size_t size_dim_4_;
    std::size_t height_;
    std::size_t width_;
    std::size_t depth_;
};

inline shape5 make_shape5_with(
    const shape5& default_shape,
    const shape5_variable shape)
{
    return shape5(
        fplus::just_with_default(default_shape.size_dim_5_, shape.size_dim_5_),
        fplus::just_with_default(default_shape.size_dim_4_, shape.size_dim_4_),
        fplus::just_with_default(default_shape.height_, shape.height_),
        fplus::just_with_default(default_shape.width_, shape.width_),
        fplus::just_with_default(default_shape.depth_, shape.depth_));
}

inline bool operator == (const shape5& lhs, const shape5_variable& rhs)
{
    return
        (rhs.size_dim_5_.is_nothing() || lhs.size_dim_5_ == rhs.size_dim_5_.unsafe_get_just()) &&
        (rhs.size_dim_4_.is_nothing() || lhs.size_dim_4_ == rhs.size_dim_4_.unsafe_get_just()) &&
        (rhs.height_.is_nothing() || lhs.height_ == rhs.height_.unsafe_get_just()) &&
        (rhs.width_.is_nothing() || lhs.width_ == rhs.width_.unsafe_get_just()) &&
        (rhs.depth_.is_nothing() || lhs.depth_ == rhs.depth_.unsafe_get_just());
}

inline bool operator == (const std::vector<shape5>& lhss,
    const std::vector<shape5_variable>& rhss)
{
    return fplus::all(fplus::zip_with(
        [](const shape5& lhs, const shape5_variable& rhs) -> bool
        {
            return lhs == rhs;
        },
        lhss, rhss));
}

inline bool operator == (const shape5& lhs, const shape5& rhs)
{
    return
        lhs.size_dim_5_ == rhs.size_dim_5_ &&
        lhs.size_dim_4_ == rhs.size_dim_4_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const shape5& lhs, const shape5& rhs)
{
    return !(lhs == rhs);
}

inline shape5 dilate_shape5(
    const shape2& dilation_rate, const shape5& s)
{
    assertion(dilation_rate.height_ >= 1, "invalid dilation rate");
    assertion(dilation_rate.width_ >= 1, "invalid dilation rate");

    const std::size_t height = s.height_ +
        (s.height_ - 1) * (dilation_rate.height_ - 1);
    const std::size_t width = s.width_ +
        (s.width_ - 1) * (dilation_rate.width_ - 1);
    return shape5(s.size_dim_5_, s.size_dim_4_, height, width, s.depth_);
}

} // namespace internal

using shape5 = internal::shape5;

inline std::string show_shape5(const shape5& s)
{
    const std::vector<std::size_t> dimensions = {
        s.size_dim_5_,
        s.size_dim_4_,
        s.height_,
        s.width_,
        s.depth_
        };
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions);
}

inline std::string show_shape5s(
    const std::vector<shape5>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape5, shapes));
}

} // namespace fdeep
