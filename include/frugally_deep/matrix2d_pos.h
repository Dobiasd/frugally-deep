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

class tensor2_pos
{
public:
    explicit tensor2_pos(
        std::size_t y,
        std::size_t x) :
            y_(y),
            x_(x)
    {
    }

    std::size_t y_;
    std::size_t x_;
};

inline bool operator == (const tensor2_pos& lhs, const tensor2_pos& rhs)
{
    return
        lhs.y_ == rhs.y_ &&
        lhs.x_ == rhs.x_;
}

inline std::string show_tensor2_pos(const tensor2_pos& pos)
{
    return std::string(
        "(" + std::to_string(pos.y_) +
        "," + std::to_string(pos.x_) +
        ")");
}

} // namespace fdeep
