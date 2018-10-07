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

class tensor3_pos_yxz
{
public:
    explicit tensor3_pos_yxz(
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            y_(y),
            x_(x),
            z_(z)
    {
    }

    std::size_t y_;
    std::size_t x_;
    std::size_t z_;
};

inline bool operator == (const tensor3_pos_yxz& lhs, const tensor3_pos_yxz& rhs)
{
    return
        lhs.y_ == rhs.y_ &&
        lhs.x_ == rhs.x_ &&
        lhs.z_ == rhs.z_;
}

} } // namespace fdeep, namespace internal
