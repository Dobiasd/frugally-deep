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
#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4706)
#pragma warning(disable : 4996)
#endif
#include <Eigen/Core>
#if defined _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#include <fplus/fplus.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

#if defined(__GNUC__) || defined(__GNUG__)
#define FDEEP_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define FDEEP_FORCE_INLINE __forceinline
#else
#define FDEEP_FORCE_INLINE inline
#endif

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

#ifdef FDEEP_FLOAT_TYPE
    typedef FDEEP_FLOAT_TYPE float_type;
#else
    typedef float float_type;
#endif

#if EIGEN_VERSION_AT_LEAST(3,3,0)
    typedef Eigen::Index EigenIndex;
#else
    typedef Eigen::DenseIndex EigenIndex;
#endif

typedef std::vector<float_type> float_vec_unaligned;

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

typedef aligned_vector<float_type> float_vec;
typedef fplus::shared_ref<float_vec> shared_float_vec;

using ColMajorMatrixXf = Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using RowMajorMatrixXf = Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MappedColMajorMatrixXf = Eigen::Map<ColMajorMatrixXf, Eigen::Aligned>;
using MappedRowMajorMatrixXf = Eigen::Map<RowMajorMatrixXf, Eigen::Aligned>;

} } // namespace fdeep, namespace internal
