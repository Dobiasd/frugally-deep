// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/global_pooling_layer.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class global_max_pooling_3d_layer : public global_pooling_layer
{
public:
    explicit global_max_pooling_3d_layer(const std::string& name) :
    global_pooling_layer(name)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    {        
        return in; // todo
    }
};

} } // namespace fdeep, namespace internal
