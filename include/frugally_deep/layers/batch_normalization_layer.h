// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

class batch_normalization_layer : public layer
{
public:
    explicit batch_normalization_layer(const size3d& size_in, float_t epsilon)
        : layer(size_in, size_in),
        epsilon_(epsilon)
    {
    }
    std::size_t param_count() const override
    {
        // todo: not implemented yet
        return 0;
    }
    float_vec get_params() const override
    {
        // todo: not implemented yet
        return {};
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        // todo: not implemented yet
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
    }
    void random_init_params() override
    {
        // todo: not implemented yet
    }
protected:
    float_t epsilon_;
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        //std::size_t n = input.size().volume();
        //float_t mu = fplus::sum(input.as_vector()) / static_cast<float_t>(n);

        input.size().volume();

        // todo: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        return matrix3d(size3d(1,1,1));
    }

    matrix3d backward_pass_impl(const matrix3d& fb,
        float_vec&) const override
    {
        // todo
        fb.size().volume();
        return matrix3d(size3d(1,1,1));
    }
};

} // namespace fd
