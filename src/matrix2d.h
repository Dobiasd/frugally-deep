#pragma once

#include "size2d.h"

#include <cstddef>
#include <string>
#include <vector>

class matrix2d
{
public:
    explicit matrix2d(const size2d& size) :
        size_(size),
        values_(size.area(), 0.0f)
    {
    }
    float get(std::size_t y, size_t x) const
    {
        return values_[idx(y, x)];
    }
    void set(std::size_t y, size_t x, float value)
    {
        values_[idx(y, x)] = value;
    }
    const size2d& size() const
    {
        return size_;
    }

private:
    std::size_t idx(std::size_t y, size_t x) const
    {
        return size().width() + y * size().width() + x;
    };
    size2d size_;
    std::vector<float> values_;
};

inline std::string show_matrix2d(const matrix2d& m)
{
    std::string str;
    str += "[";
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            str += std::to_string(m.get(y, x)) + ",";
        }
        str += "]\n";
    }
    str += "]";
    return str;
}
