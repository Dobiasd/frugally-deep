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

class node;
using node_ptr = std::shared_ptr<node>;
using node_ptrs = std::vector<node_ptr>;

node_ptr get_node(const node_ptrs& pool, const node_connection& connection);

class node
{
public:
    explicit node(
        const std::string& outbound_layer_id,
        std::size_t idx,
        const node_connections& inbound_nodes) :
            outbound_layer_id_(outbound_layer_id),
            idx_(idx),
            inbound_connections_(inbound_nodes),
            outputs_()
    {
    }
    matrix3d get_output(const layer_ptrs& layers,
        const node_ptrs& node_pool,
        std::size_t tensor_idx) const
    {
        if (fplus::is_nothing(outputs_))
            calculate_outputs(layers, node_pool);
        const auto outputs = fplus::throw_on_nothing(
            fd::error("no output values"), outputs_);
        assertion(tensor_idx < outputs.size(), "invalid tensor index");
        return outputs[tensor_idx];
    }
    void set_outputs(const matrix3ds& outputs)
    {
        // todo: only for nodes of input-layers!
        outputs_ = fplus::just<matrix3ds>(outputs);
    }
    std::string outbound_layer_id_;
    std::size_t idx_;

private:
    node_connections inbound_connections_;
    void calculate_outputs(const layer_ptrs& layers,
        const node_ptrs& node_pool) const
    {
        assertion(fplus::is_nothing(outputs_), "outputs already calculated");
        matrix3ds inputs;
        // todo: as transform
        for (std::size_t i = 0; i < inbound_connections_.size(); ++i)
        {
            inputs.push_back(get_node(
                node_pool, inbound_connections_[i])->get_output(layers,
                    node_pool, inbound_connections_[i].tensor_idx_));
        }
        outputs_ = get_layer(layers, outbound_layer_id_)->apply(inputs);
    }
    mutable fplus::maybe<matrix3ds> outputs_;
};

inline bool is_node_connected(const node_connection& conn,
    const node_ptr& target)
{
    return
        conn.layer_id_ == target->outbound_layer_id_ &&
        conn.node_idx_ == target->idx_;
}

inline node_ptr get_node(const node_ptrs& pool,
    const node_connection& connection)
{
    return fplus::throw_on_nothing(
        fd::error("dangling node connection"),
        fplus::find_first_by(
            fplus::bind_1st_of_2(is_node_connected, connection),
            pool));
}

} // namespace fd
