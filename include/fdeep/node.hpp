// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fdeep { namespace internal
{

struct node_connection
{
    node_connection(const std::string& layer_id,
        std::size_t node_idx,
        std::size_t tensor_idx) :
            layer_id_(layer_id), node_idx_(node_idx), tensor_idx_(tensor_idx)
    {}
    std::pair<std::string, std::size_t> without_tensor_idx() const
    {
        return std::make_pair(layer_id_, node_idx_);
    }
    std::string layer_id_;
    std::size_t node_idx_;
    std::size_t tensor_idx_;
};
using node_connections = std::vector<node_connection>;

using output_dict = std::map<std::pair<std::string, std::size_t>, tensors>;

class layer;
typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;
layer_ptr get_layer(const layer_ptrs& layers, const std::string& layer_id);
tensor get_layer_output(const layer_ptrs& layers, output_dict& output_cache,
    const layer_ptr& layer, std::size_t node_idx, std::size_t tensor_idx);
tensors apply_layer(const layer& layer, const tensors& inputs);

class node
{
public:
    explicit node(const node_connections& inbound_nodes) :
            inbound_connections_(inbound_nodes)
    {
    }
    tensors get_output(const layer_ptrs& layers, output_dict& output_cache,
        const layer& layer) const
    {
        const auto get_input = [&output_cache, &layers]
            (const node_connection& conn) -> tensor
        {
            return get_layer_output(layers, output_cache,
                get_layer(layers, conn.layer_id_),
                conn.node_idx_, conn.tensor_idx_);
        };
        return apply_layer(layer,
            fplus::transform(get_input, inbound_connections_));
    }
private:
    node_connections inbound_connections_;
};

typedef std::vector<node> nodes;

} } // namespace fdeep, namespace internal
