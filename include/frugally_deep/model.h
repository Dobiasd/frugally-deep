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

    explicit model(const std::string& name,
        const layer_ptrs& layers,
        const node_connections& input_connections,
        const node_connections& output_connections)
            : layer(name),
            layers_(layers),
            input_connections_(input_connections),
            output_connections_(output_connections)
    {
        // todo: make sure layers elems are unique
    }
    virtual ~model()
    {
    }
    matrix3ds predict(const matrix3ds& inputs) const
    {
        return apply_impl(inputs);
    }

    matrix3d get_output(const layer_ptrs& layers,
        output_dict& output_cache,
        std::size_t node_idx, std::size_t tensor_idx) const override
    {
        // https://stackoverflow.com/questions/46011749/understanding-keras-model-architecture-node-index-of-nested-model
        node_idx = node_idx - 1;
        assertion(node_idx < nodes_.size(), "invalid node index");

        // todo: eigene layer appenden?
        return layer::get_output(layers, output_cache, node_idx, tensor_idx);

        // todo: append raus und nur layers_?
        //return nodes_[node_idx].get_output(fplus::append(layers, layers_),
          //  output_cache, *this, tensor_idx);
    }

protected:
    virtual matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        output_dict output_cache;

        assertion(inputs.size() == input_connections_.size(),
            "invalid number of input tensors for this model: " +
            fplus::show(input_connections_.size()) + " required but " +
            fplus::show(inputs.size()) + " provided");

        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            output_cache[input_connections_[i]] = {inputs[i]};
        }

        // todo: as transform
        matrix3ds outputs;
        for (std::size_t i = 0; i < output_connections_.size(); ++i)
        {
            outputs.push_back(get_layer(
                layers_, output_connections_[i].layer_id_)->get_output(
                    layers_,
                    output_cache,
                    output_connections_[i].node_idx_,
                    output_connections_[i].tensor_idx_));
        }

        return outputs;
    }
    layer_ptrs layers_;
    node_connections input_connections_;
    node_connections output_connections_;
};

} // namespace fd
