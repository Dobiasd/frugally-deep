// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

class convolutional_layer : public layer
{
public:
    static std::vector<filter> generate_filters(
        std::size_t depth, const size2d& filter_size, std::size_t k)
    {
        return std::vector<filter>(k, filter(matrix3d(
            size3d(depth, filter_size.height_, filter_size.width_)), 0));
    }
    explicit convolutional_layer(
            const size3d& size_in, const size2d& filter_size,
            std::size_t k, std::size_t stride)
        : layer(size_in, size3d(k, size_in.height_, size_in.width_)),
        filters_(generate_filters(size_in.depth_, filter_size, k))
    {
        assert(stride == 1); // todo: allow different strides
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const filter& f) { return f.param_count(); },
            filters_);
        return fplus::sum(counts);
    }
    float_vec get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const filter& f) { return f.get_params(); },
                filters_));
    }
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
        auto params_per_filter =
            fplus::split_every(filters_.front().param_count(), params);
        for (std::size_t i = 0; i < filters_.size(); ++i)
        {
            filters_[i].set_params(params_per_filter[i]);
        }
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return convolve(filters_, input);
    }
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec& params_deltas_acc) const override
    {
        const auto output = convolve(flip_filters_spatially(filters_), input);

        float_vec params_deltas(param_count(), 0);

        // see slide 10 of
        // http://de.slideshare.net/kuwajima/cnnbp

        //float_vec params_deltas = convolve(last_input_, input).as_vector();

/*
        for (std::size_t inc = 0; inc < last_input_.size().depth_; inc++)
        {
            for (std::size_t outc = 0; outc < input.size().depth_; outc++)
            {
                for (cnn_size_t wy = 0; wy < weight_.height_; wy++)
                {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++)
                    {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
                        const float_t * delta = &curr_delta[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
                        }
                        dW[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        }
*/

        params_deltas_acc = fplus::append(params_deltas, params_deltas_acc);
        return output;
    }
    filter_vec filters_;
};

} // namespace fd
