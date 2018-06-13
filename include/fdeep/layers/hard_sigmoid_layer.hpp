// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <algorithm>
#include <string>

namespace fdeep { namespace internal
{

class hard_sigmoid_layer : public activation_layer
{
public:
    explicit hard_sigmoid_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    static float_type activation_function(float_type x)
    {
        return static_cast<float_type>(
            std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
    };
    tensor3 transform_input(const tensor3& in_vol) const override
    {
        return transform_tensor3(activation_function, in_vol);
    }
};

} } // namespace fdeep, namespace internal
