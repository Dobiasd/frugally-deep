// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace fdeep { namespace internal
{

class shape2_variable
{
public:
    explicit shape2_variable(
        fplus::maybe<std::size_t> height,
        fplus::maybe<std::size_t> width) :
            height_(height),
            width_(width)
    {
    }

    fplus::maybe<std::size_t> height_;
    fplus::maybe<std::size_t> width_;
};

} } // namespace fdeep, namespace internal
