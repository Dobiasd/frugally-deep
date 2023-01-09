// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class dot_layer : public layer
{
public:
    explicit dot_layer(const std::string& name)
        : layer(name)
    {
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        assertion(input.size() == 2,
            "need exactly 2 input tensors");
        return {dot_product_tensors(input[0], input[1])};
    }
};

} } // namespace fdeep, namespace internal
