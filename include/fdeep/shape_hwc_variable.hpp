// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape_hw_variable.hpp"
#include "fdeep/shape_hw.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class shape_hwc_variable
{
public:
    explicit shape_hwc_variable(
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

    shape_hw_variable without_depth() const
    {
        return shape_hw_variable(height_, width_);
    }

    fplus::maybe<std::size_t> height_;
    fplus::maybe<std::size_t> width_;
    fplus::maybe<std::size_t> depth_;
};

inline bool operator == (const shape_hwc_variable& lhs, const shape_hwc_variable& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const shape_hwc_variable& lhs, const shape_hwc_variable& rhs)
{
    return !(lhs == rhs);
}

} // namespace internal

using shape_hwc_variable = internal::shape_hwc_variable;

inline std::string show_shape_hwc_variable(const shape_hwc_variable& s)
{
    const std::vector<fplus::maybe<std::size_t>> dimensions =
        {s.height_, s.width_, s.depth_};
    const auto dimensions_repr = fplus::transform(
        fplus::show_maybe<std::size_t>, dimensions);
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions_repr);
}

inline std::string show_shape_hwcs_variable(
    const std::vector<shape_hwc_variable>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape_hwc_variable, shapes));
}

} // namespace fdeep
