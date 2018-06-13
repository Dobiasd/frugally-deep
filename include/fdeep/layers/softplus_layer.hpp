// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class softplus_layer : public activation_layer
{
public:
    explicit softplus_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor3 transform_input(const tensor3& in_vol) const override
    {
        auto activation_function = [](float_type x) -> float_type
        {
            // https://github.com/tensorflow/tensorflow/blob/626808e4e4a83aafbb3809a30db57bb78e839040/tensorflow/core/kernels/softplus_op.h#L41
            const float_type threshold =
                std::log(std::numeric_limits<float_type>::epsilon()) + 2;
            if (x > -threshold) // too_large
                return x;
            else if (x < threshold) // too_small
                return std::exp(x);
            else
                return std::log1p(std::exp(x));
        };
        return transform_tensor3(activation_function, in_vol);
    }
};

} } // namespace fdeep, namespace internal
