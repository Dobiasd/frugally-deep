// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace fd
{

class size2d
{
public:
    explicit size2d(
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

inline bool operator == (const size2d& lhs, const size2d& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

inline bool operator != (const size2d& lhs, const size2d& rhs)
{
    return !(lhs == rhs);
}
inline std::string show_size2d(const size2d& size)
{
    return std::string(
        "(" + std::to_string(size.height_) +
        "," + std::to_string(size.width_) +
        ")");
}

} // namespace fd
