// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2_variable.hpp"
#include "fdeep/shape2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class shape3_variable
{
public:
    explicit shape3_variable(
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

    shape2_variable without_depth() const
    {
        return shape2_variable(height_, width_);
    }

    fplus::maybe<std::size_t> height_;
    fplus::maybe<std::size_t> width_;
    fplus::maybe<std::size_t> depth_;
};

inline bool operator == (const shape3_variable& lhs, const shape3_variable& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const shape3_variable& lhs, const shape3_variable& rhs)
{
    return !(lhs == rhs);
}

} // namespace internal

using shape3_variable = internal::shape3_variable;

inline std::string show_shape3_variable(const shape3_variable& s)
{
    const std::vector<fplus::maybe<std::size_t>> dimensions =
        {s.height_, s.width_, s.depth_};
    const auto dimensions_repr = fplus::transform(
        fplus::show_maybe<std::size_t>, dimensions);
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions_repr);
}

inline std::string show_shape3s_variable(
    const std::vector<shape3_variable>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape3_variable, shapes));
}

} // namespace fdeep
