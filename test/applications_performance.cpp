// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "fdeep/fdeep.hpp"

int main()
{
    std::vector<std::string> model_paths = {
        "test_model_small.json",
        "test_model_sequential.json",
        "test_model_full.json",
        "inceptionv3.json",
        "resnet50.json",
        "vgg16.json",
        "vgg19.json",
        "xception.json",
        "densenet201.json",
        "nasnetlarge.json",
        //"inceptionvresnetv2.json", // lambda
        //"mobilenet.json", // relu6
    };

    bool error = false;

    for (const auto& model_path : model_paths)
    {
        std::cout << model_path << std::endl;
        #ifdef NDEBUG
        try
        {
            const auto model = fdeep::load_model(model_path, true);
        }
        catch (const std::exception& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl;
            error = true;
        }
        #else
            const auto model = fdeep::load_model(model_path, true);
        #endif
    }

    if (error)
    {
        std::cout << "There were errors." << std::endl;
        return 1;
    }
    std::cout << "All imports and test OK." << std::endl;
}
