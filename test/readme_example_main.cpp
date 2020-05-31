// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <fdeep/fdeep.hpp>

TEST_CASE("readme_example_main, main")
{
    const auto model = fdeep::load_model("../readme_example_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),
        fdeep::float_vec{1, 2, 3, 4})});
    std::cout << fdeep::show_tensors(result) << std::endl;
}
