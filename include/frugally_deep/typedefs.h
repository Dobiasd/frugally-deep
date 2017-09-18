// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

namespace fd
{

typedef double float_t;
typedef std::vector<float_t> float_vec;
typedef float_vec::const_iterator float_vec_const_it;
typedef float_vec::iterator float_vec_it;
typedef std::vector<float_vec> float_vecs;
const float_t pi = static_cast<float_t>(std::acos(-1));

void assertion(bool cond, const std::string& error)
{
    if (!cond)
    {
        throw std::runtime_error(error);
    }
}

} // namespace fd
