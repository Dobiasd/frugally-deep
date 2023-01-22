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

class global_max_pooling_1d_layer : public global_pooling_layer
{
public:
    explicit global_max_pooling_1d_layer(const std::string& name) :
    global_pooling_layer(name)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {
        tensor out(tensor_shape(in.shape().depth_), 0);
        for (std::size_t z = 0; z < in.shape().depth_; ++z)
        {
            float_type val = std::numeric_limits<float_type>::lowest();
            for (std::size_t x = 0; x < in.shape().width_; ++x)
            {
                val = std::max(val, in.get_ignore_rank(tensor_pos(x, z)));
            }
            out.set_ignore_rank(tensor_pos(z), val);
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
