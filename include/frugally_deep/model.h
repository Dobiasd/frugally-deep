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
    using node_id = std::string;
    using node_ids = std::vector<node_id>;
    using node_dict = std::unordered_map<node_id, node>;
    node_dict nodes_;
    explicit model(const std::string& name,
        const node_dict& nodes,
        const node_ids& inputs,
        const node_ids& outputs)
            : layer(name),
            nodes_(nodes),
            inputs_(inputs),
            outputs_(outputs)
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
    node_ids inputs_;
    node_ids outputs_;
};

} // namespace fd
