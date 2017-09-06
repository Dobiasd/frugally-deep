// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"

#include <cstddef>
#include <memory>

namespace fd
{

class node
{
public:
    explicit node(const std::string& name)
        : layer(name)
    {
    }
    virtual ~node()
    {
    }
    virtual matrix3d apply(const matrix3d& input)
    {
        // todo
        return input;
    }
};

} // namespace fd
