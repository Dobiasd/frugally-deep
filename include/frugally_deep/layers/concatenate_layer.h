// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class concatenate_layer : public layer
{
public:
    explicit concatenate_layer(const std::string& name)
        : layer(name)
    {
    }
protected:
    matrix3ds apply_impl(const matrix3ds& input) const override
    {
        return {concatenate_matrix3ds(input)};
    }
};

} // namespace fd
