// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"

#include "frugally_deep/layers/layer.h"

#include <cstddef>
#include <memory>

namespace fd
{

class model : public layer
{
public:
    // todo: nodes keine shared pointers mehr sondern normaler vector
    static nodes generate_nodes(const node_connections& connections)
    {
        // todo: transform
        nodes result;
        for (const auto& conn : connections)
        {
            result.push_back(node(node_connections({conn})));
        }
        return result;
    };
    explicit model(const std::string& name,
        const layer_ptrs& layers,
        const node_connections& input_connections,
        const node_connections& output_connections)
            : layer(name),
            layers_(layers),
            input_connections_(input_connections),
            output_connections_(output_connections)
    {
        set_nodes(generate_nodes(output_connections));
        // todo: make sure layers elems are unique
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

        // todo: as transform
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            get_layer(layers_, input_connections_[i].layer_id_)->set_outputs(
                input_connections_[i].node_idx_, {inputs[i]});
        }

        // todo: as transform
        for (std::size_t i = 0; i < output_connections_.size(); ++i)
        {
            const auto node_outputs = get_layer(
                layers_, output_connections_[i].layer_id_)->get_output(
                    layers_,
                    output_connections_[i].node_idx_,
                    output_connections_[i].tensor_idx_);
        }

        // todo
        return inputs;
    }
    layer_ptrs layers_;
    node_connections input_connections_;
    node_connections output_connections_;
};

} // namespace fd
