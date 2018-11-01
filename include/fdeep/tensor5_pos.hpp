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
    // The dimensions are right-aligned (left-padded) compared to Keras.
    // I.e., if you have a position (or shape) of (a, b) in Keras
    // it corresponds to (0, 0, 0, a, b) in frugally-deep.
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

} } // namespace fdeep, namespace internal
