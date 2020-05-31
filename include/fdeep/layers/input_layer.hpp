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

class input_layer : public layer
{
public:
    explicit input_layer(const std::string& name, const tensor_shape_variable& input_shape)
        : layer(name), input_shape_(input_shape), output_()
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override
    {
        assertion(inputs.size() == 1, "need exactly one input");
        assertion(inputs.front().shape() == input_shape_, "invalid input size");
        return inputs;
    }
    tensor_shape_variable input_shape_;

    // provide initial tensor for computation
    mutable fplus::maybe<tensor> output_;
};

} } // namespace fdeep, namespace internal
