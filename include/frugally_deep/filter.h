// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"
#include "frugally_deep/size3d.h"

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class filter
{
public:
    filter(const matrix3d& m, float_t bias) : m_(m), bias_(bias)
    {
        assert(m_.size().width_ %2 == 1);
        assert(m_.size().height_ %2 == 1);
    }
    std::size_t param_count() const
    {
        return m_.size().volume() + 1; // +1 for bias
    }
    const size3d& size() const
    {
        return m_.size();
    }
    matrix3d get_matrix3d() const
    {
        return m_;
    }
    float_t get(std::size_t z, std::size_t y, size_t x) const
    {
        return m_.get(z, y, x);
    }
    float_t get_bias() const
    {
        return bias_;
    }
    float_vec get_params() const
    {
        float_vec params;
        params.reserve(param_count());
        for (std::size_t z = 0; z < m_.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < m_.size().height_; ++y)
            {
                for (std::size_t x = 0; x < m_.size().width_; ++x)
                {
                    params.push_back(m_.get(z, y, x));
                }
            }
        }
        params.push_back(bias_);
        return params;
    }
    void set_params(const float_vec& params)
    {
        assert(params.size() == param_count());
        std::size_t i = 0;
        for (std::size_t z = 0; z < m_.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < m_.size().height_; ++y)
            {
                for (std::size_t x = 0; x < m_.size().width_; ++x)
                {
                    m_.set(z, y, x, params[i++]);
                }
            }
        }
        bias_ = params[i++];
    }
private:
    matrix3d m_;
    float_t bias_;
};

} // namespace fd
