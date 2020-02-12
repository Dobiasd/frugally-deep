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

class tensor_pos
{
public:
    // The dimensions are right-aligned (left-padded) compared to Keras.
    // I.e., if you have a position (or shape) of (a, b) in Keras
    // it corresponds to (0, 0, 0, a, b) in frugally-deep.
    explicit tensor_pos(
        std::size_t pos_dim_5,
        std::size_t pos_dim_4,
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            rank_(5),
            pos_dim_5_(pos_dim_5),
            pos_dim_4_(pos_dim_4),
            y_(y),
            x_(x),
            z_(z)
    {
    }

    explicit tensor_pos(
        std::size_t pos_dim_4,
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            rank_(4),
            pos_dim_5_(0),
            pos_dim_4_(pos_dim_4),
            y_(y),
            x_(x),
            z_(z)
    {
    }

    explicit tensor_pos(
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            rank_(3),
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(y),
            x_(x),
            z_(z)
    {
    }

    explicit tensor_pos(
        std::size_t x,
        std::size_t z) :
            rank_(2),
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(0),
            x_(x),
            z_(z)
    {
    }

    explicit tensor_pos(
        std::size_t z) :
            rank_(1),
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(0),
            x_(0),
            z_(z)
    {
    }

    std::size_t rank_;
    std::size_t pos_dim_5_;
    std::size_t pos_dim_4_;
    std::size_t y_;
    std::size_t x_;
    std::size_t z_;
};

inline std::size_t get_tensor_pos_dimension_by_index(const tensor_pos& p,
    const std::size_t idx)
{
    assertion(p.rank_ >= 4 - idx, "Invalid index for rank");
    if (idx == 0)
        return p.pos_dim_5_;
    if (idx == 1)
        return p.pos_dim_4_;
    if (idx == 2)
        return p.y_;
    if (idx == 3)
        return p.x_;
    if (idx == 4)
        return p.z_;
    raise_error("Invalid tensor_pos index.");
    return 0;
}

inline tensor_pos change_tensor_pos_dimension_by_index(const tensor_pos& in,
    const std::size_t idx, const std::size_t dim)
{
    assertion(in.rank_ >= 4 - idx, "Invalid index for rank");
    tensor_pos out = in;
    if (idx == 0)
        out.pos_dim_5_ = dim;
    else if (idx == 1)
        out.pos_dim_4_ = dim;
    else if (idx == 2)
        out.y_ = dim;
    else if (idx == 3)
        out.x_ = dim;
    else if (idx == 4)
        out.z_ = dim;
    else
        raise_error("Invalid tensor_pos index.");
    return out;
}

} } // namespace fdeep, namespace internal
