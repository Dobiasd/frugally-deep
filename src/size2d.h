#pragma once

#include "typedefs.h"

#include <cstddef>
#include <cstdlib>

class size2d
{
public:
    explicit size2d(
        std::size_t height,
        std::size_t width) :
            height_(height),
            width_(width)
    {
    }
    std::size_t height() const
    {
        return height_;
    }
    std::size_t width() const
    {
        return width_;
    }
    std::size_t area() const
    {
        return height() * width();
    }

private:
    std::size_t height_;
    std::size_t width_;
};

bool operator == (const size2d& lhs, const size2d& rhs);
