// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"

#include <cstddef>
#include <memory>

namespace fd
{

class layer
{
public:
    layer(const size3d& size_in, const size3d& size_out)
        : size_in_(size_in),
        size_out_(size_out),
        last_input_(size_in_)
    {

    }
    virtual ~layer()
    {
    }
    virtual matrix3d forward_pass(const matrix3d& input) final
    {
        assert(input.size() == input_size());
        auto output = forward_pass_impl(input);
        assert(output.size() == output_size());
        last_input_ = input;
        return output;
    }
    virtual matrix3d backward_pass(const matrix3d& input,
        float_vec& params_deltas_acc_reversed) final
    {
        std::size_t params_deltas_acc_size_before = params_deltas_acc_reversed.size();
        assert(input.size() == output_size());
        auto output = backward_pass_impl(input, params_deltas_acc_reversed);
        assert(output.size() == input_size());
        assert(params_deltas_acc_reversed.size() ==
            params_deltas_acc_size_before + param_count());
        last_input_ = input;
        return output;
    }
    virtual std::size_t param_count() const = 0;
    virtual float_vec get_params() const = 0;
    virtual void set_params(const float_vec& params) final
    {
        set_params(std::begin(params), std::end(params));
    }
    virtual void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) = 0;
    virtual const size3d& input_size() const final
    {
        return size_in_;
    }
    virtual const size3d& output_size() const final
    {
        return size_out_;
    }

protected:
    const size3d size_in_;
    const size3d size_out_;
    matrix3d last_input_;
    virtual matrix3d forward_pass_impl(const matrix3d& input) const = 0;
    virtual matrix3d backward_pass_impl(const matrix3d&, float_vec&) const = 0;
};

typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;

} // namespace fd
