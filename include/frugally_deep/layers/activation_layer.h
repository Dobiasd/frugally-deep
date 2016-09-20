// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// Abstract base class for actication layers
// https://en.wikipedia.org/wiki/Activation_function
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
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
    }
    void random_init_params() override
    {
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

    virtual matrix3d transform_error_backward_pass(const matrix3d&) const = 0;
};

} // namespace fd
