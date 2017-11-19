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

using eigen_mat = Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic>;
using eigen_idx = Eigen::Index;

shared_float_vec eigen_mat_to_values(const eigen_mat& m)
{
    shared_float_vec result = fplus::make_shared_ref<float_vec>();
    result->reserve(static_cast<std::size_t>(m.rows() * m.cols()));
    for (Eigen::Index y = 0; y < m.rows(); ++y)
    {
        for (Eigen::Index x = 0; x < m.cols(); ++x)
        {
            result->push_back(m(y, x));
        }
    }
    return result;
}

eigen_mat eigen_mat_from_values(std::size_t height, std::size_t width,
    const float_vec& values)
{
    assertion(height * width == values.size(), "invalid shape");
    eigen_mat m(height, width);
    std::size_t i = 0;
    for (Eigen::Index y = 0; y < m.rows(); ++y)
    {
        for (Eigen::Index x = 0; x < m.cols(); ++x)
        {
            m(y, x) = values[i++];
        }
    }
    return m;
}

} } // namespace fdeep, namespace internal
