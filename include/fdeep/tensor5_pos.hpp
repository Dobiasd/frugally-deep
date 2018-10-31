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

class tensor5_pos
{
public:
    explicit tensor5_pos(
        std::size_t pos_dim_5,
        std::size_t pos_dim_4,
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            pos_dim_5_(pos_dim_5),
            pos_dim_4_(pos_dim_4),
            y_(y),
            x_(x),
            z_(z)
    {
    }

    std::size_t pos_dim_5_;
    std::size_t pos_dim_4_;
    std::size_t y_;
    std::size_t x_;
    std::size_t z_;
};

inline bool operator == (const tensor5_pos& lhs, const tensor5_pos& rhs)
{
    return
        lhs.pos_dim_5_ == rhs.pos_dim_5_ &&
        lhs.pos_dim_4_ == rhs.pos_dim_4_ &&
        lhs.y_ == rhs.y_ &&
        lhs.x_ == rhs.x_ &&
        lhs.z_ == rhs.z_;
}

} } // namespace fdeep, namespace internal
