// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <cstddef>
#include <memory>

namespace fd
{

struct node_connection
{
    node_connection(const std::string& layer_id,
        std::size_t node_idx,
        std::size_t tensor_idx) :
            layer_id_(layer_id), node_idx_(node_idx), tensor_idx_(tensor_idx)
    {}
    std::string layer_id_;
    std::size_t node_idx_;
    std::size_t tensor_idx_;
};
using node_connections = std::vector<node_connection>;

class node
{
public:
    explicit node(
        const std::string& outbound_layer_id,
        std::size_t idx,
        const node_connections& inbound_nodes) :
            outbound_layer_id_(outbound_layer_id),
            idx_(idx),
            inbound_nodes_(inbound_nodes),
            outputs_()
    {
    }
    matrix3d get_output(std::size_t) const
    {
        // todo size_t parameter is tensor_idx
        return matrix3d(size3d(0,0,0), {});
    }
    void set_outputs(const matrix3ds& outputs)
    {
        // todo: only for nodes of input-layers?
        outputs_ = fplus::just<matrix3ds>(outputs);
    }
private:
    std::string outbound_layer_id_;
    std::size_t idx_;
    node_connections inbound_nodes_;
    fplus::maybe<matrix3ds> outputs_;
};

using nodes = std::vector<node>;

} // namespace fd
