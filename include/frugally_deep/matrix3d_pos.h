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

class matrix3d_pos
{
public:
    explicit matrix3d_pos(
        std::size_t z,
        std::size_t y,
        std::size_t x) :
            z_(z),
            y_(y),
            x_(x)
    {
    }
    std::size_t z() const
    {
        return z_;
    }
    std::size_t y() const
    {
        return y_;
    }
    std::size_t x() const
    {
        return x_;
    }

private:
    std::size_t z_;
    std::size_t y_;
    std::size_t x_;
};

inline bool operator == (const matrix3d_pos& lhs, const matrix3d_pos& rhs)
{
    return
        lhs.z() == rhs.z() &&
        lhs.y() == rhs.y() &&
        lhs.x() == rhs.x();
}

std::string show_matrix3d_pos(const matrix3d_pos& pos)
{
    return std::string(
        "(" + std::to_string(pos.z()) +
        "," + std::to_string(pos.y()) +
        "," + std::to_string(pos.x()) +
        ")");
}

} // namespace fd
