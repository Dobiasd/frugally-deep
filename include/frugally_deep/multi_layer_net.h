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

struct input_with_output
{
    matrix3d input_;
    matrix3d output_;
};
typedef std::vector<input_with_output> input_with_output_vec;

class multi_layer_net : public layer
{
public:
    explicit multi_layer_net(const std::vector<layer_ptr>& layer_ptrs) :
        layer_ptrs_(layer_ptrs)
    {
        assert(is_layer_ptr_chain_valid(layer_ptrs_));
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        auto current_volume = input;
        for (const auto& layer_ptr : layer_ptrs_)
        {
            current_volume = layer_ptr->forward_pass(current_volume);
        }
        return current_volume;
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const layer_ptr& l) { return l->param_count(); },
            layer_ptrs_);
        return fplus::sum(counts);
    }
    float_vec get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const layer_ptr& l) { return l->get_params(); },
                layer_ptrs_));
    }
    void set_params(const float_vec& params) override
    {
        auto layer_param_counts = fplus::transform(
            [](const layer_ptr& l) { return l->param_count(); },
            layer_ptrs_);
        auto split_idxs =
            fplus::scan_left_1(std::plus<std::size_t>(), layer_param_counts);
        auto params_per_layer = fplus::split_at_idxs(split_idxs, params);
        for (std::size_t i = 0; i < layer_ptrs_.size(); ++i)
        {
            layer_ptrs_[i]->set_params(params_per_layer[i]);
        }
    }
    std::size_t input_depth() const override
    {
        return layer_ptrs_.front()->input_depth();
    }
    std::size_t output_depth() const override
    {
        return layer_ptrs_.back()->output_depth();
    }

private:
    static bool is_layer_transition_valid(
        const std::pair<layer_ptr, layer_ptr>& layer_ptr_pair)
    {
        // todo
        return true;
        return
            layer_ptr_pair.first->output_depth() ==
            layer_ptr_pair.second->input_depth();
    }
    static bool is_layer_ptr_chain_valid(
        const std::vector<layer_ptr>& layer_ptrs)
    {
        // todo: what about non-conv-layers like FC?
        return
            fplus::all_by(
                [](const layer_ptr& ptr) -> bool
                {
                    return static_cast<bool>(ptr);
                },
                layer_ptrs) &&
            fplus::all_by(
                is_layer_transition_valid,
                fplus::overlapping_pairs(layer_ptrs));
    }
    std::vector<layer_ptr> layer_ptrs_;
};

} // namespace fd
