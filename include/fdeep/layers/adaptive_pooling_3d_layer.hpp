// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace fdeep {
namespace internal {

    enum class adaptive_pooling_kind {
        avg,
        max
    };

    class adaptive_pooling_3d_layer : public layer {
    public:
        explicit adaptive_pooling_3d_layer(const std::string& name,
            std::size_t out_d4, std::size_t out_h, std::size_t out_w,
            adaptive_pooling_kind kind)
            : layer(name)
            , out_d4_(out_d4)
            , out_h_(out_h)
            , out_w_(out_w)
            , kind_(kind)
        {
        }

    protected:
        std::size_t out_d4_;
        std::size_t out_h_;
        std::size_t out_w_;
        adaptive_pooling_kind kind_;

        static std::size_t adapt_start(std::size_t i, std::size_t in_size, std::size_t out_size)
        {
            return static_cast<std::size_t>(std::floor(
                static_cast<double>(i * in_size) / static_cast<double>(out_size)));
        }

        static std::size_t adapt_end(std::size_t i, std::size_t in_size, std::size_t out_size)
        {
            return static_cast<std::size_t>(std::ceil(
                static_cast<double>((i + 1) * in_size) / static_cast<double>(out_size)));
        }

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto& sh = input.shape();
            const std::size_t in_d4 = sh.size_dim_4_;
            const std::size_t in_h = sh.height_;
            const std::size_t in_w = sh.width_;
            const std::size_t depth = sh.depth_;
            const std::size_t out_d4 = out_d4_ == 0 ? in_d4 : out_d4_;
            const std::size_t out_h = out_h_ == 0 ? in_h : out_h_;
            const std::size_t out_w = out_w_;

            tensor_shape out_shape(out_d4, out_h, out_w, depth);
            // Match input rank when possible (so 1D/2D inputs produce 1D/2D outputs).
            if (sh.rank() <= 3) {
                if (sh.rank() == 2)
                    out_shape = tensor_shape(out_w, depth);
                else
                    out_shape = tensor_shape(out_h, out_w, depth);
            } else if (sh.rank() == 4) {
                out_shape = tensor_shape(out_d4, out_h, out_w, depth);
            }

            tensor out(out_shape, float_type(0));

            for (std::size_t od = 0; od < out_d4; ++od) {
                const std::size_t d_start = in_d4 == 1 ? 0 : adapt_start(od, in_d4, out_d4);
                const std::size_t d_end = in_d4 == 1 ? 1 : adapt_end(od, in_d4, out_d4);
                for (std::size_t oy = 0; oy < out_h; ++oy) {
                    const std::size_t y_start = in_h == 1 ? 0 : adapt_start(oy, in_h, out_h);
                    const std::size_t y_end = in_h == 1 ? 1 : adapt_end(oy, in_h, out_h);
                    for (std::size_t ox = 0; ox < out_w; ++ox) {
                        const std::size_t x_start = adapt_start(ox, in_w, out_w);
                        const std::size_t x_end = adapt_end(ox, in_w, out_w);
                        for (std::size_t z = 0; z < depth; ++z) {
                            float_type acc = kind_ == adaptive_pooling_kind::max
                                ? std::numeric_limits<float_type>::lowest()
                                : float_type(0);
                            std::size_t count = 0;
                            for (std::size_t d = d_start; d < d_end; ++d) {
                                for (std::size_t y = y_start; y < y_end; ++y) {
                                    for (std::size_t x = x_start; x < x_end; ++x) {
                                        const float_type v = input.get_ignore_rank(
                                            tensor_pos(0, d, y, x, z));
                                        if (kind_ == adaptive_pooling_kind::max)
                                            acc = std::max(acc, v);
                                        else
                                            acc += v;
                                        ++count;
                                    }
                                }
                            }
                            if (kind_ == adaptive_pooling_kind::avg && count > 0)
                                acc /= static_cast<float_type>(count);
                            out.set_ignore_rank(tensor_pos(0, od, oy, ox, z), acc);
                        }
                    }
                }
            }

            return { out };
        }
    };

}
}
