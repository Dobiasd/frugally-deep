#pragma once

#include "matrix3d.h"
#include "size3d.h"

#include <cstddef>
#include <vector>

class filter
{
public:
    explicit filter(const matrix3d& m) : m_(m)
    {
    }
    std::size_t param_count() const
    {
        return m_.size().area();
    }
    const size3d& size() const
    {
        return m_.size();
    }
    matrix3d get_matrix3d() const
    {
        return m_;
    }
    float get(std::size_t z, std::size_t y, size_t x) const
    {
        return m_.get(z, y, x);
    }
    std::vector<float> get_params() const;
    void set_params(const std::vector<float>& params);

private:
    matrix3d m_;
};
