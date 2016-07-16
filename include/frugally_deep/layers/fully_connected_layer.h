// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include "frugally_deep/matrix2d.h"

namespace fd
{

class fully_connected_layer : public layer
{
public:
    fully_connected_layer(std::size_t n_in, std::size_t n_out)
        : params_(size2d(n_in, n_out))
    {
    }

    matrix3d forward_pass(const matrix3d& input) const override
    {
        // todo
        return input;
    }
    std::size_t param_count() const override
    {
        return params_.size().height() * params_.size().width();
    }
    float_vec get_params() const override
    {
        // todo params_ flatten
        return {};
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
        // todo
    }
    std::size_t input_depth() const override
    {
        // todo: everything goes
        // perhaps instead fplus::maybe<std::size_t> fixed_input_depth()
        return 0;
    }
    std::size_t output_depth() const override
    {
        // todo: same_as_input
        return 0;
    }
private:
    matrix2d params_;
};

} // namespace fd
