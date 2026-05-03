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
        const std::size_t out_d4_;
        const std::size_t out_h_;
        const std::size_t out_w_;
        const adaptive_pooling_kind kind_;

        struct range {
            std::size_t start;
            std::size_t end;
        };

        // Adaptive pooling maps output index i to input range
        // [floor(i * in / out), ceil((i+1) * in / out)). For length-1 input
        // dimensions (which can occur for promoted 1D/2D tensors) the range
        // collapses to [0, 1).
        static range adapt_range(std::size_t i, std::size_t in_size, std::size_t out_size)
        {
            if (in_size == 1)
                return { 0, 1 };
            assertion(out_size > 0, "AdaptivePooling output_size must be > 0.");
            const auto start = static_cast<std::size_t>(std::floor(
                static_cast<double>(i * in_size) / static_cast<double>(out_size)));
            const auto end = static_cast<std::size_t>(std::ceil(
                static_cast<double>((i + 1) * in_size) / static_cast<double>(out_size)));
            return { start, end };
        }

        static tensor_shape output_shape_for(const tensor_shape& in_shape,
            std::size_t out_d4, std::size_t out_h, std::size_t out_w)
        {
            const std::size_t depth = in_shape.depth_;
            switch (in_shape.rank()) {
            case 2:
                return tensor_shape(out_w, depth);
            case 3:
                return tensor_shape(out_h, out_w, depth);
            default:
                return tensor_shape(out_d4, out_h, out_w, depth);
            }
        }

        float_type pool_window(const tensor& input,
            range d, range h, range w, std::size_t z) const
        {
            const bool is_max = kind_ == adaptive_pooling_kind::max;
            float_type acc = is_max
                ? std::numeric_limits<float_type>::lowest()
                : float_type(0);
            std::size_t count = 0;
            for (std::size_t di = d.start; di < d.end; ++di) {
                for (std::size_t yi = h.start; yi < h.end; ++yi) {
                    for (std::size_t xi = w.start; xi < w.end; ++xi) {
                        const float_type v = input.get_ignore_rank(tensor_pos(0, di, yi, xi, z));
                        acc = is_max ? std::max(acc, v) : acc + v;
                        ++count;
                    }
                }
            }
            if (!is_max && count > 0)
                acc /= static_cast<float_type>(count);
            return acc;
        }

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto& sh = input.shape();
            const std::size_t out_d4 = out_d4_ == 0 ? sh.size_dim_4_ : out_d4_;
            const std::size_t out_h = out_h_ == 0 ? sh.height_ : out_h_;
            const std::size_t out_w = out_w_;

            tensor out(output_shape_for(sh, out_d4, out_h, out_w), float_type(0));

            for (std::size_t od = 0; od < out_d4; ++od) {
                const range d = adapt_range(od, sh.size_dim_4_, out_d4);
                for (std::size_t oy = 0; oy < out_h; ++oy) {
                    const range h = adapt_range(oy, sh.height_, out_h);
                    for (std::size_t ox = 0; ox < out_w; ++ox) {
                        const range w = adapt_range(ox, sh.width_, out_w);
                        for (std::size_t z = 0; z < sh.depth_; ++z) {
                            out.set_ignore_rank(tensor_pos(0, od, oy, ox, z),
                                pool_window(input, d, h, w, z));
                        }
                    }
                }
            }

            return { out };
        }
    };

}
}
