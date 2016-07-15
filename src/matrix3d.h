#pragma once

#include "typedefs.h"

#include "size3d.h"

#include <cstddef>
#include <string>
#include <vector>

class matrix3d
{
public:
    explicit matrix3d(const size3d& size) :
        size_(size),
        values_(size.area(), 0.0f)
    {
    }
    float_t get(std::size_t z, std::size_t y, std::size_t x) const
    {
        return values_[idx(z, y, x)];
    }
    void set(std::size_t z, std::size_t y, size_t x, float_t value)
    {
        values_[idx(z, y, x)] = value;
    }
    const size3d& size() const
    {
        return size_;
    }

private:
    std::size_t idx(std::size_t z, std::size_t y, size_t x) const
    {
        return z * size().height() * size().width() + y * size().width() + x;
    };
    size3d size_;
    float_vec values_;
};

inline std::string show_matrix3d(const matrix3d& m)
{
    std::string str;
    str += "[";
    for (std::size_t z = 0; z < m.size().depth(); ++z)
    {
        str += "[";
        for (std::size_t y = 0; y < m.size().height(); ++y)
        {
            for (std::size_t x = 0; x < m.size().width(); ++x)
            {
                str += std::to_string(m.get(z, y, x)) + ",";
            }
            str += "]\n";
        }
        str += "]\n";
    }
    str += "]";
    return str;
}
