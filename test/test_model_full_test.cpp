// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <fdeep/fdeep.hpp>

TEST_CASE("test_model_full_test, load_model")
{
    const auto model = fdeep::load_model("keras_export/test_model_small.json");
}
