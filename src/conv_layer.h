#pragma once

#include "typedefs.h"

#include "convolution.h"
#include "filter.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <vector>

class conv_layer : public layer
{
public:
    explicit conv_layer(const std::vector<filter>& filters) :
        filters_(filters)
    {
        assert(fplus::is_not_empty(filters));
        auto filter_sizes =
            fplus::transform([](const filter& f) { return f.size(); },
            filters_);
        assert(fplus::all_the_same(filter_sizes));
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return convolve(filters_, input);
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
    std::size_t input_depth() const override
    {
        return filters_.front().get_matrix3d().size().depth();
    }
    std::size_t output_depth() const override
    {
        return filters_.size();
    }
private:
    std::vector<filter> filters_;
};
