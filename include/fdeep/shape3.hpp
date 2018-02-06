// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2.hpp"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace fdeep { namespace internal
{

class shape3
{
public:
    explicit shape3(
        std::size_t depth,
        std::size_t height,
        std::size_t width) :
            depth_(depth),
            height_(height),
            width_(width)
    {
    }
    std::size_t volume() const
    {
        return depth_ * height_ * width_;
    }

    shape2 without_depth() const
    {
        return shape2(height_, width_);
    }

    std::size_t depth_;
    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const shape3& lhs, const shape3& rhs)
{
    return
        lhs.depth_ == rhs.depth_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

inline bool operator != (const shape3& lhs, const shape3& rhs)
{
    return !(lhs == rhs);
}

inline shape3 dilate_shape3(const shape2& dilation_rate, const shape3& s)
{
    assertion(dilation_rate.height_ >= 1, "invalid dilation rate");
    assertion(dilation_rate.width_ >= 1, "invalid dilation rate");

    const std::size_t height = s.height_ +
        (s.height_ - 1) * (dilation_rate.height_ - 1);
    const std::size_t width = s.width_ +
        (s.width_ - 1) * (dilation_rate.width_ - 1);
    return shape3(s.depth_, height, width);
}

} // namespace internal

using shape3 = internal::shape3;

} // namespace fdeep
