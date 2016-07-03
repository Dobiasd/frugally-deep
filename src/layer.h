#pragma once

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
    virtual std::vector<float> get_params() const = 0;
    virtual void set_params(const std::vector<float>& params) = 0;
    virtual std::size_t input_depth() const = 0;
    virtual std::size_t output_depth() const = 0;
};

typedef std::shared_ptr<layer> layer_ptr;
