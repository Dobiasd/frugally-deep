// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/pooling_3d_layer.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace fdeep { namespace internal
{

FDEEP_FORCE_INLINE tensor max_pool_3d(
    std::size_t pool_size_d4, std::size_t pool_height, std::size_t pool_width,
    std::size_t strides_d4, std::size_t strides_y, std::size_t strides_x,
    padding, //pad_type,
    const tensor& in)
{   
    std::cout << pool_size_d4 << pool_height << pool_width << strides_d4 << strides_y << strides_x << std::endl;
    return in; // todo
}

class max_pooling_3d_layer : public pooling_3d_layer
{
public:
    explicit max_pooling_3d_layer(const std::string& name,
        const shape3& pool_size, const shape3& strides,
        padding p) :
        pooling_3d_layer(name, pool_size, strides, p)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {
        return max_pool_3d(
            pool_size_.depth_, pool_size_.height_, pool_size_.width_,
            strides_.depth_, strides_.height_, strides_.width_,
            padding_, in);
    }
};

} } // namespace fdeep, namespace internal
