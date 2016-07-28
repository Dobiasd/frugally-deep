// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class multi_layer_net : public layer
{
public:
    explicit multi_layer_net(const std::vector<layer_ptr>& layers) :
        layer(
            layers.empty() || !layers.front()
                ? size3d(0,0,0)
                : layers.front()->input_size(),
            layers.empty() || !layers.back()
                ? size3d(0,0,0)
                : layers.back()->output_size()),
        layers_(layers)
    {
        assert(is_layer_ptr_chain_valid(layers));
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const layer_ptr& l) { return l->param_count(); },
            layers_);
        return fplus::sum(counts);
    }
    float_vec get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const layer_ptr& l) { return l->get_params(); },
                layers_));
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
        auto it = ps_begin;
        for (auto& layer : layers_)
        {
            const auto layer_param_count =
                static_cast<float_vec::difference_type>(layer->param_count());
            layer->set_params(it, it + layer_param_count);
            it += layer_param_count;
        }
    }
    void random_init_params() override
    {
        for (auto& layer : layers_)
        {
            layer->random_init_params();
        }
    }

protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        auto current_volume = input;
        for (const auto& current_layer : layers_)
        {
            current_volume = current_layer->forward_pass(current_volume);
        }
        return current_volume;
    }
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec& params_deltas_acc) const override
    {
        auto current_volume = input;
        const auto layers_reversed = fplus::reverse(layers_);
        for (const auto& current_layer : layers_reversed)
        {
            current_volume = current_layer->backward_pass(
                current_volume, params_deltas_acc);
        }
        return current_volume;
    }
    static bool is_layer_transition_valid(
        const std::pair<layer_ptr, layer_ptr>& layer_ptr_pair)
    {
        return
            layer_ptr_pair.first->output_size() ==
            layer_ptr_pair.second->input_size();
    }
    static bool is_layer_ptr_chain_valid(
        const std::vector<layer_ptr>& layers)
    {
        if (!fplus::all_by(
                [](const layer_ptr& ptr) -> bool
                {
                    return static_cast<bool>(ptr);
                },
                layers))
        {
            // todo exception mit beschreibung
            return false;
        }

        // todo exception mit beschreibung
        return fplus::all_by(
                is_layer_transition_valid,
                fplus::overlapping_pairs(layers));
    }
    std::vector<layer_ptr> layers_;
};

} // namespace fd
