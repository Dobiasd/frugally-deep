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

class shape5_variable
{
public:
    explicit shape5_variable(
        fplus::maybe<std::size_t> size_dim_5,
        fplus::maybe<std::size_t> size_dim_4,
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            size_dim_5_(size_dim_5),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

    fplus::maybe<std::size_t> size_dim_5_;
    fplus::maybe<std::size_t> size_dim_4_;
    fplus::maybe<std::size_t> height_;
    fplus::maybe<std::size_t> width_;
    fplus::maybe<std::size_t> depth_;
};

inline bool operator == (const shape5_variable& lhs, const shape5_variable& rhs)
{
    return
        lhs.size_dim_5_ == rhs.size_dim_5_ &&
        lhs.size_dim_4_ == rhs.size_dim_4_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const shape5_variable& lhs, const shape5_variable& rhs)
{
    return !(lhs == rhs);
}

} // namespace internal

using shape5_variable = internal::shape5_variable;

inline std::string show_shape5_variable(const shape5_variable& s)
{
    const std::vector<fplus::maybe<std::size_t>> dimensions = {
        s.size_dim_5_,
        s.size_dim_4_,
        s.height_,
        s.width_,
        s.depth_
        };
    const auto dimensions_repr = fplus::transform(
        fplus::show_maybe<std::size_t>, dimensions);
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions_repr);
}

inline std::string show_shape5s_variable(
    const std::vector<shape5_variable>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape5_variable, shapes));
}

} // namespace fdeep
