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
            pos_dim_5_(pos_dim_5),
            pos_dim_4_(pos_dim_4),
            y_(y),
            x_(x),
            z_(z),
            rank_(5)
    {
    }

    explicit tensor_pos(
        std::size_t pos_dim_4,
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            pos_dim_5_(0),
            pos_dim_4_(pos_dim_4),
            y_(y),
            x_(x),
            z_(z),
            rank_(4)
    {
    }

    explicit tensor_pos(
        std::size_t y,
        std::size_t x,
        std::size_t z) :
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(y),
            x_(x),
            z_(z),
            rank_(3)
    {
    }

    explicit tensor_pos(
        std::size_t x,
        std::size_t z) :
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(0),
            x_(x),
            z_(z),
            rank_(2)
    {
    }

    explicit tensor_pos(
        std::size_t z) :
            pos_dim_5_(0),
            pos_dim_4_(0),
            y_(0),
            x_(0),
            z_(z),
            rank_(1)
    {
    }

    std::size_t rank() const
    {
        return rank_;
    }

    std::vector<std::size_t> dimensions() const
    {
        if (rank() == 5)
            return {pos_dim_5_, pos_dim_4_, y_, x_, z_};
        if (rank() == 4)
            return {pos_dim_4_, y_, x_, z_};
        if (rank() == 3)
            return {y_, x_, z_};
        if (rank() == 2)
            return {x_, z_};
        return {z_};
    }

    std::size_t pos_dim_5_;
    std::size_t pos_dim_4_;
    std::size_t y_;
    std::size_t x_;
    std::size_t z_;

private:
    std::size_t rank_;
};

inline tensor_pos create_tensor_pos_from_dims(
    const std::vector<std::size_t>& dimensions)
{
    assertion(dimensions.size() >= 1 && dimensions.size() <= 5,
        "Invalid tensor-pos dimensions");
    if (dimensions.size() == 5)
        return tensor_pos(
            dimensions[0],
            dimensions[1],
            dimensions[2],
            dimensions[3],
            dimensions[4]);
    if (dimensions.size() == 4)
        return tensor_pos(
            dimensions[0],
            dimensions[1],
            dimensions[2],
            dimensions[3]);
    if (dimensions.size() == 3)
        return tensor_pos(
            dimensions[0],
            dimensions[1],
            dimensions[2]);
    if (dimensions.size() == 2)
        return tensor_pos(
            dimensions[0],
            dimensions[1]);
    return tensor_pos(dimensions[0]);
}

inline tensor_pos tensor_pos_with_changed_rank(const tensor_pos& s, std::size_t rank)
{
    assertion(rank >= 1 && rank <= 5, "Invalid target rank");
    if (rank == 4)
    {
        assertion(s.pos_dim_5_ == 0, "Invalid target rank");
        return tensor_pos(s.pos_dim_4_, s.y_, s.x_, s.z_);
    }
    if (rank == 3)
    {
        assertion(s.pos_dim_5_ == 0, "Invalid target rank");
        assertion(s.pos_dim_4_ == 0, "Invalid target rank");
        return tensor_pos(s.y_, s.x_, s.z_);
    }
    if (rank == 2)
    {
        assertion(s.pos_dim_5_ == 0, "Invalid target rank");
        assertion(s.pos_dim_4_ == 0, "Invalid target rank");
        assertion(s.y_ == 0, "Invalid target rank");
        return tensor_pos(s.x_, s.z_);
    }
    if (rank == 1)
    {
        assertion(s.pos_dim_5_ == 0, "Invalid target rank");
        assertion(s.pos_dim_4_ == 0, "Invalid target rank");
        assertion(s.y_ == 0, "Invalid target rank");
        assertion(s.x_ == 0, "Invalid target rank");
        return tensor_pos(s.z_);
    }
    return tensor_pos(s.pos_dim_5_, s.pos_dim_4_, s.y_, s.x_, s.z_);
}

} // namespace internal

using tensor_pos = internal::tensor_pos;

} // namespace fdeep
