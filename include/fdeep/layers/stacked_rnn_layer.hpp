// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    // Wraps a chain of recurrent layers built from a StackedRNNCells. Each
    // inner layer is a standalone LSTM / GRU / SimpleRNN where all but the
    // last run with return_sequences=true so the next cell sees the full
    // sequence at every timestep.
    class stacked_rnn_layer : public layer {
    public:
        explicit stacked_rnn_layer(const std::string& name,
            const std::vector<layer_ptr>& inner_layers)
            : layer(name)
            , inner_layers_(inner_layers)
        {
        }

    protected:
        const std::vector<layer_ptr> inner_layers_;

        tensors apply_impl(const tensors& inputs) const override
        {
            tensors current = inputs;
            for (const auto& inner : inner_layers_)
                current = inner->apply(current);
            return current;
        }
    };

}
}
