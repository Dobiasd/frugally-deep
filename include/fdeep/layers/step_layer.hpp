// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

namespace fdeep { namespace internal
{

class step_layer : public activation_layer
{
public:
    explicit step_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor3 transform_input(const tensor3& in_vol) const override
    {
        const auto activation_function = [this](float_type x) -> float_type
        {
            return x > 0 ? static_cast<float_type>(1) :
                static_cast<float_type>(0);
        };
        return transform_tensor3(activation_function, in_vol);
    }
};

} } // namespace fdeep, namespace internal
