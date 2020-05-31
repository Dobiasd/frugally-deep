// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <string>

namespace fdeep { namespace internal
{

// https://arxiv.org/pdf/1706.02515.pdf
class selu_layer : public activation_layer
{
public:
    explicit selu_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    const float_type alpha_ =
        static_cast<float_type>(1.6732632423543772848170429916717);
    const float_type scale_ =
        static_cast<float_type>(1.0507009873554804934193349852946);
    tensor transform_input(const tensor& in_vol) const override
    {
        return transform_tensor(selu_activation, in_vol);
    }
};

} } // namespace fdeep, namespace internal
