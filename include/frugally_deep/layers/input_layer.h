// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class input_layer : public layer
{
public:
    explicit input_layer(const std::string& name, const size3d& input_size)
        : layer(name), input_size_(input_size), output_()
    {
    }
    void set_output(const matrix3d& output) const override
    {
        output_ = output;
    }
    virtual matrix3d get_output(const layer_ptrs&,
        std::size_t node_idx, std::size_t tensor_idx) const
    {
        assertion(node_idx == 0, "invalid node index for input layer");
        assertion(tensor_idx == 0, "invalid tensor index for input layer");
        return fplus::throw_on_nothing(
            fd::error("no input tensor deposited"), output_);
    }
    bool is_input_layer() const override
    {
        return true;
    }
protected:
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assertion(inputs.size() == 1, "need exactly one input");
        assertion(inputs[0].size() == input_size_, "invalid input size");
        return inputs;
    }
    size3d input_size_;

    // provide initial tensor for computation
    mutable fplus::maybe<matrix3d> output_;
};

} // namespace fd
