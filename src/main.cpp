// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "frugally_deep/frugally_deep.hpp"

int main()
{
    const auto model = fdeep::load_model("keras_export/model.json", true);
    //const auto xception = fdeep::load_model("keras_export/xception.json");
    //const auto vgg16 = fdeep::load_model("keras_export/vgg16.json");
    //const auto vgg19 = fdeep::load_model("keras_export/vgg19.json", true);
    //const auto resnet50 = fdeep::load_model("keras_export/resnet50.json");
    //const auto vgginceptionv39 = fdeep::load_model("keras_export/inceptionv3.json");
    //const auto inceptionvresnetv2 = fdeep::load_model("keras_export/inceptionvresnetv2.json");
    //const auto mobilenet = fdeep::load_model("keras_export/mobilenet.json");
    std::cout << "done" << std::endl;
}
