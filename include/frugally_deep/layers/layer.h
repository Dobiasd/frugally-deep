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

class layer
{
public:
    explicit layer(const std::string& name)
        : name_(name)
    {
    }
    virtual ~layer()
    {
    }
    virtual matrix3d apply(const matrix3d& input) const = 0;
    virtual const std::string& name() const final
    {
        return name_;
    }

protected:
    const std::string& name_;
};

typedef std::shared_ptr<layer> layer_ptr;
typedef std::vector<layer_ptr> layer_ptrs;

} // namespace fd
