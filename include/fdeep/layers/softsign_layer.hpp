// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class softsign_layer : public activation_layer
{
public:
    explicit softsign_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor transform_input(const tensor& in_vol) const override
    {
        return transform_tensor(softsign_activation, in_vol);
    }
};

} } // namespace fdeep, namespace internal
