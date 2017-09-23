// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "frugally_deep/frugally_deep.hpp"

int main()
{
    const auto model = fdeep::load_model("keras_export/model.json");
}
