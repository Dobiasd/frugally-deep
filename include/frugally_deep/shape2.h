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

namespace fdeep
{

class shape2
{
public:
    explicit shape2(
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

inline bool operator == (const shape2& lhs, const shape2& rhs)
{
    return
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

inline bool operator != (const shape2& lhs, const shape2& rhs)
{
    return !(lhs == rhs);
}
inline std::string show_shape2(const shape2& size)
{
    return std::string(
        "(" + std::to_string(size.height_) +
        "," + std::to_string(size.width_) +
        ")");
}

inline shape2 swap_shape2_dims(const shape2& size)
{
    return shape2(size.width_, size.height_);
}

} // namespace fdeep
