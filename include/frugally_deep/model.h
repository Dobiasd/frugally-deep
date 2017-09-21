// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"

#include "frugally_deep/node.h"

#include <cstddef>
#include <memory>

namespace fd
{

class model : public layer
{
public:
    explicit model(const std::string& name,
        const layer_ptrs& layers,
        const nodes& node_pool,
        const node_connections& inputs,
        const node_connections& outputs)
            : layer(name),
            layers_(layers),
            node_pool_(node_pool),
            inputs_(inputs),
            outputs_(outputs)
    {
    }
    virtual ~model()
    {
    }
protected:
    virtual matrix3ds apply_impl(const matrix3ds& input) const override
    {
        // todo
        return input;
    }
    layer_ptrs layers_;
    nodes node_pool_;
    node_connections inputs_;
    node_connections outputs_;
};

} // namespace fd
