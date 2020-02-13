// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class tensor_shape_variable
{
public:
    explicit tensor_shape_variable(
        fplus::maybe<std::size_t> size_dim_5,
        fplus::maybe<std::size_t> size_dim_4,
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            rank_(5),
            size_dim_5_(size_dim_5),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape_variable(
        fplus::maybe<std::size_t> size_dim_4,
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            rank_(4),
            size_dim_5_(1),
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape_variable(
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            rank_(3),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(height),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape_variable(
        fplus::maybe<std::size_t> width,
        fplus::maybe<std::size_t> depth) :
            rank_(2),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(1),
            width_(width),
            depth_(depth)
    {
    }

        explicit tensor_shape_variable(
        fplus::maybe<std::size_t> depth) :
            rank_(1),
            size_dim_5_(1),
            size_dim_4_(1),
            height_(1),
            width_(1),
            depth_(depth)
    {
    }

    std::size_t rank_;
    fplus::maybe<std::size_t> size_dim_5_;
    fplus::maybe<std::size_t> size_dim_4_;
    fplus::maybe<std::size_t> height_;
    fplus::maybe<std::size_t> width_;
    fplus::maybe<std::size_t> depth_;
};

inline bool operator == (const tensor_shape_variable& lhs, const tensor_shape_variable& rhs)
{
    return
        lhs.size_dim_5_ == rhs.size_dim_5_ &&
        lhs.size_dim_4_ == rhs.size_dim_4_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

inline bool operator != (const tensor_shape_variable& lhs, const tensor_shape_variable& rhs)
{
    return !(lhs == rhs);
}

} // namespace internal

using tensor_shape_variable = internal::tensor_shape_variable;

inline std::string show_tensor_shape_variable(const tensor_shape_variable& s)
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
    return std::to_string(s.rank()) + fplus::show_cont_with_frame(", ", "(", ")", dimensions_repr);
}

inline std::string show_tensor_shapes_variable(
    const std::vector<tensor_shape_variable>& shapes)
{
    return fplus::show_cont(fplus::transform(show_tensor_shape_variable, shapes));
}

} // namespace fdeep
