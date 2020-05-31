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

class global_max_pooling_2d_layer : public global_pooling_layer
{
public:
    explicit global_max_pooling_2d_layer(const std::string& name, bool channels_first) :
    global_pooling_layer(name, channels_first)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {
        const std::size_t feature_count = channels_first_
            ? in.shape().height_
            : in.shape().depth_
            ;

        const std::size_t in_height = channels_first_
            ? in.shape().width_
            : in.shape().height_
            ;

        const std::size_t in_width = channels_first_
            ? in.shape().depth_
            : in.shape().width_
            ;

        tensor out(tensor_shape(feature_count), 0);
        for (std::size_t z = 0; z < feature_count; ++z)
        {
            float_type val = std::numeric_limits<float_type>::lowest();
            for (std::size_t y = 0; y < in_height; ++y)
            {
                for (std::size_t x = 0; x < in_width; ++x)
                {
                    if (channels_first_)
                        val = std::max(val, in.get_ignore_rank(tensor_pos(z, y, x)));
                    else
                        val = std::max(val, in.get_ignore_rank(tensor_pos(y, x, z)));
                }
            }
            out.set_ignore_rank(tensor_pos(z), val);
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
