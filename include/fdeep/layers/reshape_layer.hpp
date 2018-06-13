// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class reshape_layer : public layer
{
public:
    explicit reshape_layer(const std::string& name,
        const std::vector<int>& target_shape_channel_last)
        : layer(name),
        target_shape_channel_last_(target_shape_channel_last)
    {
    }
protected:
    tensor3s apply_impl(const tensor3s& input) const override
    {
        assertion(input.size() == 1,
            "reshape layer needs exactly one input tensor");
        return {reshape_tensor3(input[0], target_shape_channel_last_)};
    }
    std::vector<int> target_shape_channel_last_;
};

} } // namespace fdeep, namespace internal
