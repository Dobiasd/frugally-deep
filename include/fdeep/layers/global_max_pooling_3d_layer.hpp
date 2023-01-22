// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/global_pooling_layer.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class global_max_pooling_3d_layer : public global_pooling_layer
{
public:
    explicit global_max_pooling_3d_layer(const std::string& name) :
    global_pooling_layer(name, false)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {        
        tensor out(tensor_shape(in.shape().depth_), std::numeric_limits<float_type>::lowest());
        for (std::size_t d4 = 0; d4 < in.shape().size_dim_4_; ++d4)
        {
            for (std::size_t y = 0; y < in.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.shape().width_; ++x)
                {
                    for (std::size_t z = 0; z < in.shape().depth_; ++z)
                    {
                        out.set_ignore_rank(tensor_pos(z), std::max(
                            out.get_ignore_rank(tensor_pos(z)),
                            in.get_ignore_rank(tensor_pos(d4, y, x, z))));
                    }
                }
            }
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
