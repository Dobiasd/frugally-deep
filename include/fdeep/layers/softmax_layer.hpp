// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class softmax_layer : public activation_layer
{
public:
    explicit softmax_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor transform_input(const tensor& input) const override
    {
        return softmax(input);
    }
};

} } // namespace fdeep, namespace internal
