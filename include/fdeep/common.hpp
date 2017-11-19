// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <eigen3/Eigen/Dense>
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#include <fplus/fplus.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

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

using eigen_mat = Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using eigen_idx = Eigen::Index;

} } // namespace fdeep, namespace internal
