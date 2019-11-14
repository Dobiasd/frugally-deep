// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "fdeep/fdeep.hpp"

int test_convolution()
{
    const std::size_t k = 512;
    const std::size_t filter_height = 3;
    const std::size_t filter_width = 3;
    const std::size_t x_width = 56;
    const std::size_t x_height = 56;
    const std::size_t x_depth = 256;
    const int runs = 1;

    const fdeep::float_vec weights(filter_height * filter_width * x_depth * k, 0);
    const fdeep::float_vec bias(k, 0);

    fdeep::internal::conv_2d_layer layer(
        "test_conv_layer",
        fdeep::shape5(1, 1, filter_height, filter_width, x_depth),
        k,
        fdeep::internal::shape2(1, 1),
        fdeep::internal::padding::same,
        fdeep::internal::shape2(1, 1),
        weights, bias);

    const fdeep::tensor5 x(fdeep::shape5(1, 1, x_height, x_width, x_depth), 0);

    using namespace std::chrono;
    const auto start_time_ns = high_resolution_clock::now().time_since_epoch().count();
    float checksum = 0.0f; // to prevent compiler from optimizing everything away
    for (int run = 0; run < runs; ++run)
    {
        const auto y = layer.apply({x});
        checksum += y.front().get(0, 0, 1, 1, 1);
    }
    const auto end_time_ns = high_resolution_clock::now().time_since_epoch().count();
    const auto elapsed_ms = (end_time_ns - start_time_ns) / (runs * 1000000);
    std::cout << "checksum: " << checksum << ") elapsed_ms: " << elapsed_ms << std::endl;
    return 0;
}

int main()
{
    return test_convolution();

    std::vector<std::string> model_paths = {
        // todo: remove block
        "test_model_sequential.json",
        "test_model_exhaustive.json",
        "vgg19.json"

/*
        "densenet121.json",
        "densenet169.json",
        "densenet201.json",
        //"inceptionresnetv2.json", // lambda
        "inceptionv3.json",
        "mobilenet.json",
        "mobilenetv2.json",
        "nasnetlarge.json",
        "nasnetmobile.json",
        "resnet101.json",
        "resnet101v2.json",
        "resnet152.json",
        "resnet152v2.json",
        "resnet50.json",
        "resnet50v2.json",
        "vgg16.json",
        "vgg19.json",
        "xception.json"
        */
    };

    bool error = false;

    for (const auto& model_path : model_paths)
    {
        std::cout << "----" << std::endl;
        std::cout << model_path << std::endl;
        #ifdef NDEBUG
        try
        {
            const auto model = fdeep::load_model(model_path, true);
            const std::size_t warm_up_runs = 3;
            const std::size_t test_runs = 5;
            for (std::size_t i = 0; i < warm_up_runs; ++i)
            {
                const double duration = model.test_speed();
                std::cout << "Forward pass took "
                    << duration << " s." << std::endl;
            }
            double duration_sum = 0;
            std::cout << "Starting performance measurements." << std::endl;
            for (std::size_t i = 0; i < test_runs; ++i)
            {
                const double duration = model.test_speed();
                duration_sum += duration;
                std::cout << "Forward pass took "
                    << duration << " s." << std::endl;
            }
            const double duration_avg =
                duration_sum / static_cast<double>(test_runs);
            std::cout << "Forward pass took "
                << duration_avg << " s on average." << std::endl;
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
