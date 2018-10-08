// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace fdeep { namespace internal
{

class shape_hw
{
public:
    explicit shape_hw(
        std::size_t height,
        std::size_t width) :
            height_(height),
            width_(width)
    {
    }
    std::size_t area() const
    {
        return height_ * width_;
    }

    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const shape_hw& lhs, const shape_hw& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

} } // namespace fdeep, namespace internal