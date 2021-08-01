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

class repeat_vector_layer : public layer
{
public:
    explicit repeat_vector_layer(const std::string& name,
        std::size_t n)
        : layer(name),
        n_(n)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        assertion(input.shape().rank() == 1, "Invalid input shape for RepeatVector");
        return {tensor(
            tensor_shape(n_, input.shape().depth_),
            fplus::repeat(n_, *input.as_vector()))};
    }
    std::size_t n_;
};

} } // namespace fdeep, namespace internal
