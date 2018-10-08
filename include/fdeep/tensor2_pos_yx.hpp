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

class tensor2_pos_yx
{
public:
    explicit tensor2_pos_yx(
        std::size_t y,
        std::size_t x) :
            y_(y),
            x_(x)
    {
    }

    std::size_t y_;
    std::size_t x_;
};

} } // namespace fdeep, namespace internal
