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
        const node_ptrs& node_pool,
        const node_connections& input_connections,
        const node_connections& output_connections)
            : layer(name),
            layers_(layers),
            node_pool_(node_pool),
            input_connections_(input_connections),
            output_connections_(output_connections)
    {
    }
    virtual ~model()
    {
    }
    matrix3ds predict(const matrix3ds& inputs) const
    {
        return apply_impl(inputs);
    }
protected:
    virtual matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assertion(inputs.size() == input_connections_.size(),
            "invalid number of input tensors for this model: " +
            fplus::show(input_connections_.size()) + "required but " +
            fplus::show(inputs.size()) + "provided");
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            get_node(node_pool_, input_connections_[i])->set_outputs(
                {inputs[i]});
        }
        // todo
        return inputs;
    }
    layer_ptrs layers_;
    node_ptrs node_pool_;
    node_connections input_connections_;
    node_connections output_connections_;
};

} // namespace fd
