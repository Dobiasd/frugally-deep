// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "typedefs.h"

#include "matrix3d.h"

#include <cstddef>
#include <memory>

class layer
{
public:
    virtual ~layer()
    {
    }
    virtual matrix3d forward_pass(const matrix3d& input) const = 0;
    virtual std::size_t param_count() const = 0;
    virtual float_vec get_params() const = 0;
    virtual void set_params(const float_vec& params) = 0;
    virtual std::size_t input_depth() const = 0;
    virtual std::size_t output_depth() const = 0;
};

typedef std::shared_ptr<layer> layer_ptr;
