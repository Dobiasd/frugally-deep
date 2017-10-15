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

class tensor3_pos
{
public:
    explicit tensor3_pos(
        std::size_t z,
        std::size_t y,
        std::size_t x) :
            z_(z),
            y_(y),
            x_(x)
    {
    }

    std::size_t z_;
    std::size_t y_;
    std::size_t x_;
};

inline bool operator == (const tensor3_pos& lhs, const tensor3_pos& rhs)
{
    return
        lhs.z_ == rhs.z_ &&
        lhs.y_ == rhs.y_ &&
        lhs.x_ == rhs.x_;
}

} } // namespace fdeep, namespace internal
