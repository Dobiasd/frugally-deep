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

class layer;
typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;

class activation_layer;
typedef std::shared_ptr<activation_layer> activation_layer_ptr;
matrix3ds apply_activation_layer(const activation_layer_ptr& ptr, const matrix3ds& input);

class layer
{
public:
    explicit layer(const std::string& name)
        : name_(name), nodes_(), activation_(nullptr)
    {
    }
    virtual ~layer()
    {
    }

    void set_activation(const activation_layer_ptr& activation)
    {
        activation_ = activation;
    }

    void set_nodes(const nodes& layer_nodes)
    {
        nodes_ = layer_nodes;
    }

    virtual matrix3ds apply(const matrix3ds& input) const final
    {
        const auto result = apply_impl(input);
        if (activation_ == nullptr)
            return result;
        else
            return apply_activation_layer(activation_, result);
    }

    virtual matrix3d get_output(const layer_ptrs& layers,
        output_dict& output_cache,
        std::size_t node_idx, std::size_t tensor_idx) const
    {
        const node_connection conn(name_, node_idx, tensor_idx);

        if (!fplus::map_contains(output_cache, conn.without_tensor_idx()))
        {
            assertion(node_idx < nodes_.size(), "invalid node index");
            output_cache[conn.without_tensor_idx()] =
                nodes_[node_idx].get_output(layers, output_cache, *this);
        }

        const auto& outputs = fplus::get_from_map_unsafe(
            output_cache, conn.without_tensor_idx());

        assertion(tensor_idx < outputs.size(),
            "invalid tensor index");
        return outputs[tensor_idx];
    }

    std::string name_;
    nodes nodes_;

protected:
    virtual matrix3ds apply_impl(const matrix3ds& input) const = 0;
    activation_layer_ptr activation_;
};

inline matrix3d get_layer_output(const layer_ptrs& layers, output_dict& output_cache,
    const layer_ptr& layer,
    std::size_t node_idx, std::size_t tensor_idx)
{
    return layer->get_output(layers, output_cache, node_idx, tensor_idx);
}

inline matrix3ds apply_layer(const layer& layer, const matrix3ds& inputs)
{
    return layer.apply(inputs);
}

inline layer_ptr get_layer(const layer_ptrs& layers,
    const std::string& layer_id)
{
    const auto is_matching_layer = [layer_id](const layer_ptr& ptr) -> bool
    {
        return ptr->name_ == layer_id;
    };
    return fplus::throw_on_nothing(
        fd::error("dangling layer reference: " + layer_id),
        fplus::find_first_by(is_matching_layer, layers));
}

} // namespace fd
