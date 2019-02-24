// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/global_pooling_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class global_average_pooling_1d_layer : public global_pooling_layer
{
public:
    explicit global_average_pooling_1d_layer(const std::string& name, bool channels_first) :
    global_pooling_layer(name, channels_first)
    {
    }
protected:
    tensor5 pool(const tensor5& in) const override
    {
        const std::size_t feature_count = channels_first_
            ? in.shape().width_
            : in.shape().depth_
            ;

        const std::size_t step_count = channels_first_
            ? in.shape().depth_
            : in.shape().width_
            ;

        tensor5 out(shape5(1, 1, 1, 1, feature_count), 0);
        for (std::size_t z = 0; z < feature_count; ++z)
        {
            float_type val = 0;
            for (std::size_t x = 0; x < step_count; ++x)
            {
                if (channels_first_)
                    val += in.get(0, 0, 0, z, x);
                else
                    val += in.get(0, 0, 0, x, z);
            }
            out.set(0, 0, 0, 0, z, val / static_cast<float_type>(step_count));
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
