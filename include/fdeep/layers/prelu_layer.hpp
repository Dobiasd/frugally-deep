// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

namespace fdeep {
    namespace internal
    {

        class prelu_layer : public layer
        {
        public:
            explicit prelu_layer(const std::string& name, const float_vec& alpha) :
                layer(name), alpha_(alpha)
            {
            }
        protected:
            float_vec alpha_;
            tensor3s apply_impl(const tensor3s& input) const override
            {
                const auto my_shared_ref = fplus::make_shared_ref<std::vector<float>>(alpha_);
                fdeep::tensor3 alpha_tensor3(input[0].shape(), my_shared_ref);
                std::cout << "alpha_tensor shape"<< fdeep::show_shape3(input[0].shape())<<" values " << fdeep::show_tensor3(alpha_tensor3) << std::endl;

                fdeep::tensor3 out(input[0].shape(), 1.0f);
                for (int x = 0; x < out.shape().width_; x++)
                {
                    for (int y = 0; y < out.shape().height_; y++)
                    {
                        for (int z = 0; z < out.shape().depth_; z++)
                        {
                            if (input[0].get(z, y, x) > 0)
                            {
                                out.set(z, y, x, input[0].get(z, y, x));
                            }
                            else
                            {
                                out.set(z, y, x, alpha_tensor3.get(z, y, x) * input[0].get(z, y, x));
                            }
                            std::cout << "tensor " << alpha_tensor3.get(z, y, x) << " vector " << alpha_[z*out.shape().height_*out.shape().width_ + y * out.shape().width_ + x] << std::endl;
                        }
                    }
                }
                return { out };
            }
        };

    }
} // namespace fdeep, namespace internal
