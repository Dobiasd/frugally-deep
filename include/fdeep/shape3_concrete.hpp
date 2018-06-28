// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/shape2_concrete.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class shape3_concrete
{
public:
    explicit shape3_concrete(
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

    shape2_concrete without_depth() const
    {
        return shape2_concrete(height_, width_);
    }

    std::size_t depth_;
    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const shape3_concrete& lhs, const shape3_concrete& rhs)
{
    return
        lhs.depth_ == rhs.depth_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

inline bool operator != (const shape3_concrete& lhs, const shape3_concrete& rhs)
{
    return !(lhs == rhs);
}

inline shape3_concrete dilate_shape3_concrete(const shape2_concrete& dilation_rate, const shape3_concrete& s)
{
    assertion(dilation_rate.height_ >= 1, "invalid dilation rate");
    assertion(dilation_rate.width_ >= 1, "invalid dilation rate");

    const std::size_t height = s.height_ +
        (s.height_ - 1) * (dilation_rate.height_ - 1);
    const std::size_t width = s.width_ +
        (s.width_ - 1) * (dilation_rate.width_ - 1);
    return shape3_concrete(s.depth_, height, width);
}

} // namespace internal

using shape3_concrete = internal::shape3_concrete;

inline std::string show_shape3_concrete(const shape3_concrete& s)
{
    const std::vector<std::size_t> dimensions = {s.depth_, s.height_, s.width_};
    return fplus::show_cont_with_frame(", ", "(", ")", dimensions);
}

inline std::string show_shape3_concretes(const std::vector<shape3_concrete>& shapes)
{
    return fplus::show_cont(fplus::transform(show_shape3_concrete, shapes));
}

} // namespace fdeep
