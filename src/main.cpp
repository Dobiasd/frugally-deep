// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

int main()
{
    const auto model = fd::load_model("keras_export/model.json");
}
