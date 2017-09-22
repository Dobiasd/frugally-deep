// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

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

// todo: unordered map instead?
struct node_connection_less
{
   bool operator() (const node_connection& lhs, const node_connection& rhs) const
   {
        return
            lhs.layer_id_ < rhs.layer_id_ ||
            lhs.node_idx_ < rhs.node_idx_ ||
            lhs.tensor_idx_ < rhs.tensor_idx_;
   }
};
using output_dict = std::map<node_connection, matrix3ds, node_connection_less>;

class layer;
typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;
layer_ptr get_layer(const layer_ptrs& layers, const std::string& layer_id);
matrix3d get_layer_output(const layer_ptrs& layers, output_dict& output_cache,
    const layer_ptr& layer, std::size_t node_idx, std::size_t tensor_idx);
matrix3ds apply_layer(const layer& layer, const matrix3ds& inputs);

class node
{
public:
    explicit node(const node_connections& inbound_nodes) :
            inbound_connections_(inbound_nodes)
    {
    }
    matrix3ds get_output(const layer_ptrs& layers, output_dict& output_cache,
        const layer& layer) const
    {
        matrix3ds inputs;
        // todo: as transform
        for (std::size_t i = 0; i < inbound_connections_.size(); ++i)
        {
            inputs.push_back(get_layer_output(layers, output_cache, get_layer(
                layers, inbound_connections_[i].layer_id_),
                inbound_connections_[i].node_idx_,
                inbound_connections_[i].tensor_idx_));
        }
        return apply_layer(layer, inputs);
    }
private:
    node_connections inbound_connections_;
};

typedef std::vector<node> nodes;

} // namespace fd
