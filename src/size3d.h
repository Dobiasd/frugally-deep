#pragma once

#include "typedefs.h"

#include <cstddef>
#include <cstdlib>

class size3d
{
public:
    explicit size3d(
        std::size_t depth,
        std::size_t height,
        std::size_t width) :
            depth_(depth),
            height_(height),
            width_(width)
    {
    }
    std::size_t depth() const
    {
        return depth_;
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
        return depth() * height() * width();
    }

private:
    std::size_t depth_;
    std::size_t height_;
    std::size_t width_;
};

inline bool operator == (const size3d& lhs, const size3d& rhs)
{
    return
        lhs.depth() == rhs.depth() &&
        lhs.height() == rhs.height() &&
        lhs.width() == rhs.width();
}
