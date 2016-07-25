// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/size2d.h"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace fd
{

class size3d
{
public:
    explicit size3d(
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

    size2d without_depth() const
    {
        return size2d(height_, width_);
    }

    std::size_t depth_;
    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const size3d& lhs, const size3d& rhs)
{
    return
        lhs.depth_ == rhs.depth_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

inline bool operator != (const size3d& lhs, const size3d& rhs)
{
    return !(lhs == rhs);
}

inline std::string show_size3d(const size3d& size)
{
    return std::string(
        "(" + std::to_string(size.depth_) +
        "," + std::to_string(size.height_) +
        "," + std::to_string(size.width_) +
        ")");
}

} // namespace fd
