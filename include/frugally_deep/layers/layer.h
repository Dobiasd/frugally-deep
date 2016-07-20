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
    virtual ~layer()
    {
    }
    matrix3d forward_pass(const matrix3d& input) const
    {
        assert(input.size() == input_size());
        return forward_pass_impl(input);
    }
    virtual std::size_t param_count() const = 0;
    virtual float_vec get_params() const = 0;
    virtual void set_params(const float_vec& params) = 0;
    virtual const size3d& input_size() const = 0;
    virtual size3d output_size() const = 0;

protected:
    virtual matrix3d forward_pass_impl(const matrix3d& input) const = 0;
};

typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;

} // namespace fd
