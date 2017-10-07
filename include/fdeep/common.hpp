// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

#include <fplus/fplus.hpp>

namespace fdeep { namespace internal
{

inline std::runtime_error error(const std::string& error)
{
    return std::runtime_error(error);
}

inline void raise_error(const std::string& msg)
{
    throw error(msg);
}

inline void assertion(bool cond, const std::string& error)
{
    if (!cond)
    {
        raise_error(error);
    }
}

typedef float float_type;
typedef std::vector<float_type> float_vec;
typedef fplus::shared_ref<float_vec> shared_float_vec;

} } // namespace fdeep, namespace internal
