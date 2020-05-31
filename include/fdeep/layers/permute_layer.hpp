// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class permute_layer : public layer
{
public:
    explicit permute_layer(const std::string& name,
        const std::vector<std::size_t>& dims) :
            layer(name), dims_raw_(dims)
    {
        check_permute_tensor_dims(dims);
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        return {permute_tensor(input, dims_raw_)};
    }
    std::vector<std::size_t> dims_raw_;
};

} } // namespace fdeep, namespace internal
