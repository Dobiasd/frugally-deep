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

class shape3
{
public:
    explicit shape3(
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

    std::size_t height_;
    std::size_t width_;
    std::size_t depth_;
};

inline bool operator == (const shape3& lhs, const shape3& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_ &&
        lhs.depth_ == rhs.depth_;
}

} } // namespace fdeep, namespace internal
