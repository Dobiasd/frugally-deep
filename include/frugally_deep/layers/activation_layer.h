// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// Abstract base class for actication layers
class activation_layer : public layer
{
public:
    explicit activation_layer(const size3d& size_in) :
        layer(size_in, size_in)
    {
    }
    std::size_t param_count() const override
    {
        return 0;
    }
    float_vec get_params() const override
    {
        return {};
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return transform_input(input);
    }
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec&) const override
    {
        return transform_error_backward_pass(input);
    }
    virtual matrix3d transform_input(const matrix3d& input) const = 0;

    // todo: make pure virtual
    virtual matrix3d transform_error_backward_pass(const matrix3d&) const
    {
        // not implemented yet
        assert(false);
    }
};

} // namespace fd
