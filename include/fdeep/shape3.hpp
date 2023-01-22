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

class shape3
{
public:
    explicit shape3(
        std::size_t size_dim_4,
        std::size_t height,
        std::size_t width) :
            size_dim_4_(size_dim_4),
            height_(height),
            width_(width)
    {
    }
    std::size_t volume() const
    {
        return size_dim_4_ * height_ * width_;
    }

    std::size_t size_dim_4_;
    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const shape3& lhs, const shape3& rhs)
{
    return
        lhs.size_dim_4_ == rhs.size_dim_4_ &&
        lhs.height_ == rhs.height_ &&
        lhs.width_ == rhs.width_;
}

} } // namespace fdeep, namespace internal
