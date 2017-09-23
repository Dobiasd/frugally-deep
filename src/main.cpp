// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "frugally_deep/frugally_deep.hpp"

int main()
{
    std::vector<std::string> model_paths = {
        "keras_export/model.json",
        "keras_export/xception.json",
        "keras_export/vgg16.json",
        "keras_export/vgg19.json",
        "keras_export/resnet50.json",
        "keras_export/inceptionv3.json",
        "keras_export/inceptionvresnetv2.json",
        "keras_export/mobilenet.json"
    };

    for (const auto& model_path : model_paths)
    {
        try
        {
            const auto model = fdeep::load_model(model_path);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    std::cout << "done" << std::endl;
}
