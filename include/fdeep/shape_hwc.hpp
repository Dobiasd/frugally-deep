// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape_hw.hpp"
#include "fdeep/shape_hwc_variable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class shape_hwc
{
public:
    explicit shape_hwc(
        std::size_t height,
        std::size_t width,
        std::size_t depth) :
            height_(height),
            width_(width),
            depth_(depth)
    {
    }
    std::size_t volume() const
    {
        return height_ * width_ * depth_;
    }

    shape_hw without_depth() const
    {
        return shape_hw(height_, width_);
    }

    std::size_t height_;
    std::size_t width_;
    std::size_t depth_;
};

inline shape_hwc make_shape_hwc_with(
    const shape_hwc& default_shape,
    const shape_hwc_variable shape)
{
    return shape_hwc(
        fplus::just_with_default(default_shape.height_, shape.height_),
        fplus::just_with_default(default_shape.width_, shape.width_),
        fplus::just_with_default(default_shape.depth_, shape.depth_));
}

inline bool operator == (const shape_hwc& lhs, const shape_hwc_variable& rhs)
{
    return
        (rhs.height_.is_nothing() || lhs.height_ == rhs.height_.unsafe_get_just()) &&
        (rhs.width_.is_nothing() || lhs.width_ == rhs.width_.unsafe_get_just()) &&
        (rhs.depth_.is_nothing() || lhs.depth_ == rhs.depth_.unsafe_get_just());
}

inline bool operator == (const std::vector<shape_hwc>& lhss,
    const std::vector<shape_hwc_variable>& rhss)
{
    return fplus::all(fplus::zip_with(
        [](const shape_hwc& lhs, const shape_hwc_variable& rhs) -> bool
        {
            return lhs == rhs;
        },
        lhss, rhss));
}

inline bool operator == (const shape_hwc& lhs, const shape_hwc& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const shape_hwc& lhs, const shape_hwc& rhs)
{
    return !(lhs == rhs);
}

inline shape_hwc dilate_shape_hwc(
    const shape_hw& dilation_rate, const shape_hwc& s)
{
    assertion(dilation_rate.height_ >= 1, "invalid dilation rate");
    assertion(dilation_rate.width_ >= 1, "invalid dilation rate");

    const std::size_t height = s.height_ +
        (s.height_ - 1) * (dilation_rate.height_ - 1);
    const std::size_t width = s.width_ +
        (s.width_ - 1) * (dilation_rate.width_ - 1);
    return shape_hwc(height, width, s.depth_);
}

} // namespace internal

using shape_hwc = internal::shape_hwc;

inline std::string show_shape_hwc(const shape_hwc& s)
{
    const std::vector<std::size_t> dimensions =
        {s.height_, s.width_, s.depth_};
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions);
}

inline std::string show_shape_hwcs(
    const std::vector<shape_hwc>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape_hwc, shapes));
}

} // namespace fdeep
