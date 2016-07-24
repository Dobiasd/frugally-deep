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

class matrix2d_pos
{
public:
    explicit matrix2d_pos(
        std::size_t y,
        std::size_t x) :
            y_(y),
            x_(x)
    {
    }

    std::size_t y_;
    std::size_t x_;
};

inline bool operator == (const matrix2d_pos& lhs, const matrix2d_pos& rhs)
{
    return
        lhs.y_ == rhs.y_ &&
        lhs.x_ == rhs.x_;
}

inline std::string show_matrix2d_pos(const matrix2d_pos& pos)
{
    return std::string(
        "(" + std::to_string(pos.y_) +
        "," + std::to_string(pos.x_) +
        ")");
}

} // namespace fd
