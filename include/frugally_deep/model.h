// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/matrix3d.h"

#include "frugally_deep/node.h"

#include <cstddef>
#include <memory>

namespace fd
{

class model : public layer
{
public:
    explicit model(const std::string& name)
        : layer(name), nodes(), inputs(), outputs()
    {
    }
    virtual ~model()
    {
    }
protected:
    virtual matrix3ds apply_impl(const matrix3ds& input) const override
    {
        // todo
        return input;
    }
    using node_id = std::string;
    using node_ids = std::vector<node_id>;

    std::unordered_map<node_id, node> nodes;
    node_ids inputs;
    node_ids outputs;
};

} // namespace fd
