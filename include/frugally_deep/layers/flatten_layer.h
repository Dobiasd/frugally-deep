// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class flatten_layer : public layer
{
public:
    explicit flatten_layer(const size3d& size_in) :
            size_in_(size_in)
    {
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return reshape_matrix3d(input, output_size());
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
    const size3d& input_size() const override
    {
        return size_in_;
    }
    size3d output_size() const override
    {
        return size3d(1, 1, size_in_.volume());
    }
protected:
    size3d size_in_;
};

} // namespace fd
