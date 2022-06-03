// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "fdeep/fdeep.hpp"

int main()
{
    printMem("before fdeep::load_model");
    const auto model = fdeep::load_model("vgg19.json", true);
    printMem("after fdeep::load_model");
}
