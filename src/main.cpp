// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

// todo raus
#include <iostream>

#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

// http://stackoverflow.com/a/21802936
std::vector<unsigned char> read_file(const std::string& filename,
    std::size_t max_bytes)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    assert(file.good());

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    assert(fileSize == static_cast<std::streamoff>(30730000));
    if (max_bytes != 0)
    {
    	fileSize = static_cast<std::streamoff>(max_bytes);
    }
    file.seekg(0, std::ios::beg);

    // reserve memory and fill with zeroes
    std::vector<unsigned char> vec(static_cast<std::size_t>(fileSize), 0);

    // read the data:
    file.read(reinterpret_cast<char*>(&vec[0]), fileSize);

    return vec;
}

fd::input_with_output parse_cifar_10_bin_line(
    const std::vector<unsigned char>& vec)
{
    assert(vec.size() == 3073);
    fd::matrix3d output(fd::size3d(1, 10, 1), fd::float_vec(10, 0));
    assert(vec[0] < 10);
    output.set(0, vec[0], 0, 1);
    fd::matrix3d input(fd::size3d(3, 32, 32));
    std::size_t vec_i = 0;
    for (std::size_t z = 0; z < input.size().depth_; ++z)
    {
        for (std::size_t y = 0; y < input.size().height_; ++y)
        {
            for (std::size_t x = 0; x < input.size().width_; ++x)
            {
                input.set(input.size().depth_ - (z + 1), y, x,
                    vec[++vec_i] / static_cast<float_t>(256));
            }
        }
    }
    return {input, output};
}

fd::input_with_output_vec load_cifar_10_bin_file(const std::string& file_path,
		bool mini_version)
{
    std::size_t mini_version_img_count = 3;
	std::size_t max_bytes = mini_version ? 3073 * mini_version_img_count : 0;
    const auto bytes = read_file(file_path, max_bytes);
    const auto lines = fplus::split_every(3073, bytes);
    assert((mini_version && lines.size() == mini_version_img_count) ||
        lines.size() == 10000);
    return fplus::transform(parse_cifar_10_bin_line, lines);
}

fd::input_with_output_vec load_cifar_10_bin_training(
    const std::string& base_directory,
	bool mini_version)
{
    return fplus::concat(std::vector<fd::input_with_output_vec>({
        load_cifar_10_bin_file(base_directory + "/data_batch_1.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_2.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_3.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_4.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_5.bin", mini_version)}));
}

fd::input_with_output_vec load_cifar_10_bin_test(
    const std::string& base_directory,
	bool mini_version)
{
    return load_cifar_10_bin_file(base_directory + "/test_batch.bin", mini_version);
}

fd::classification_dataset load_cifar_10_bin(
    const std::string& base_directory,
	bool mini_version_training = false,
    bool mini_version_test = false)
{
    return {
        load_cifar_10_bin_training(base_directory, mini_version_training),
        load_cifar_10_bin_test(base_directory, mini_version_test)};
}

std::string frame_string(const std::string& str)
{
    return
        std::string(str.size(), '-') + "\n" +
        str + "\n" +
        std::string(str.size(), '-');
}

void lenna_filter_test()
{
    std::cout << frame_string("lenna_filter_test") << std::endl;
    cv::Mat img_uchar = cv::imread("test_images/lenna_512x512.png", cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);

    cv::Mat kernel_scharr_x(cv::Size(3,3), CV_32FC1, cv::Scalar(0));
    kernel_scharr_x.at<float>(0,0) =   3.0f / 32.0f;
    kernel_scharr_x.at<float>(1,0) =  10.0f / 32.0f;
    kernel_scharr_x.at<float>(2,0) =   3.0f / 32.0f;
    kernel_scharr_x.at<float>(0,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(1,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(2,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(0,2) =  -3.0f / 32.0f;
    kernel_scharr_x.at<float>(1,2) = -10.0f / 32.0f;
    kernel_scharr_x.at<float>(2,2) =  -3.0f / 32.0f;

    cv::Mat kernel_blur(cv::Size(3,3), CV_32FC1, cv::Scalar(1.0f/9.0f));

    cv::Mat filtered1;
    cv::filter2D(img, filtered1, CV_32FC3, kernel_scharr_x);
    cv::filter2D(filtered1, filtered1, -1, kernel_scharr_x);
    cv::resize(filtered1, filtered1, cv::Size(0,0), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(filtered1, filtered1, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    filtered1 = normalize_float_img(filtered1);
    filtered1 = float_img_to_uchar_img(filtered1);
    cv::imwrite("stuff/lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2Ds_via_net(img, {kernel_scharr_x, kernel_scharr_x});
    filtered2 = shrink_via_net(filtered2, 2);
    filtered2 = grow_via_net(filtered2, 2);
    filtered2 = normalize_float_img(filtered2);
    filtered2 = float_img_to_uchar_img(filtered2);
    cv::imwrite("stuff/lenna_512x512_filtered2.png", filtered2);

    std::cout << frame_string("filtered images written to ./stuff/lenna_512x512_filtered*.png") << std::endl;
}

void xor_as_net_test()
{
    std::cout << frame_string("xor_as_net_test") << std::endl;

    using namespace fd;

    input_with_output_vec xor_table =
    {
       {{size3d(1,2,1), {0, 0}}, {size3d(1,1,1), {0}}},
       {{size3d(1,2,1), {0, 1}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 0}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 1}}, {size3d(1,1,1), {0}}},
    };

    classification_dataset classifcation_data =
    {
        xor_table,
        xor_table
    };

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    pre_layers layers = {
        fc(4),
        tanh(),
        fc(4),
        tanh(),
        fc(1),
        tanh(),
        };

    pre_layers layers_min = {
        fc(2),
        tanh(),
        fc(1),
        tanh(),
        };
    float_vec layers_min_good_params =
    {
         1, 1, -1, -1,
         0.5f, -1.5f,
         1, 1,
         1.5f
    };

    auto xor_net = net(layers_min)(size3d(1, 2, 1));
    std::cout << "net.param_count() " << xor_net->param_count() << std::endl;

    //xor_net->set_params(layers_min_good_params);
    xor_net->random_init_params();
    train(xor_net, classifcation_data.training_data_, 100000, 0.1f, 0.1f);
    test(xor_net, classifcation_data.test_data_);
}

bool file_exists(const std::string& file_path)
{
    return static_cast<bool>(std::ifstream(file_path));
}

fd::matrix3d load_col_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

fd::matrix3d load_gray_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_gray_img_float_to_matrix3d(img);
}

fd::classification_dataset load_gradient_dataset(const std::string& base_dir)
{
    fd::classification_dataset classifcation_data =
    {
        {
            {load_gray_image_as_matrix3d(base_dir + "/training/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        },
        {
            {load_gray_image_as_matrix3d(base_dir + "/test/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        }
    };

    return classifcation_data;
}

void gradients_classification_test()
{
    std::cout << frame_string("gradients_classification_test") << std::endl;
    auto classifcation_data = load_gradient_dataset("test_images/datasets/classification/gradients");
    assert(!classifcation_data.training_data_.empty());
    assert(!classifcation_data.test_data_.empty());

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    using namespace fd;

    pre_layers layers = {
        conv(size2d(3, 3), 2, 1),
        elu(1),
        max_pool(32),
        flatten(),
        fc(2),
        //sigmoid(),
        softmax()
        };

    auto gradnet = net(layers)(size3d(1, 32, 32));
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    float_vec good_params =
    {
         3,  0,  -3,
        10,  0, -10,
         3,  0,  -3,
        0,
         3,  10,  3,
         0,   0,  0,
        -3, -10, -3,
        0,
        1,0,0,1,0,0
    };

    //gradnet->set_params(good_params);

    gradnet->random_init_params();
    //gradnet->set_params(fd::randomly_change_params(gradnet->get_params(), 0.3));

    train(gradnet, classifcation_data.training_data_, 1000, 0.01f, 0.1f);
    test(gradnet, classifcation_data.test_data_);
}

void cifar_10_classification_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.7);
    pre_layers layers = {
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 64, 1), activation_function,
        conv(size2d(3, 3), 64, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 128, 1), activation_function,
        conv(size2d(3, 3), 128, 1), activation_function,
        pooling_function,

        //conv(size2d(3, 3), 64, 1), elu(1),
        //conv(size2d(3, 3), 64, 1), elu(1),
        //max_pool(2),

        //conv(size2d(3, 3), 128, 1), elu(1),
        //conv(size2d(1, 1), 128, 1), elu(1),
        //max_pool(2),

        flatten(),
        fc(100),
        //tanh(true),
        fc(10),
        //tanh(true),
        softmax(),
        };

    // http://cs231n.github.io/convolutional-networks/
    pre_layers layers_simple_cs231n = {
        conv(size2d(3, 3), 12, 1),
        relu(),
        max_pool(2),
        flatten(),
        fc(10),
        softmax(),
        };

    pre_layers layers_simple = {
        conv(size2d(3, 3), 12, 1),
        elu(1),
        gentle_max_pool(2, 0.7),
        flatten(),
        fc(10),
        softmax(),
        };

    pre_layers layers_linear = {
        flatten(),
        fc(10),
        softmax(),
        };



    std::cout << frame_string("cifar_10_classification_test") << std::endl;
    std::cout << "loading cifar-10 ..." << std::flush;
    auto classifcation_data = load_cifar_10_bin(
        "./stuff/cifar-10-batches-bin", false, false);
    std::cout << " done" << std::endl;

    //classifcation_data = normalize_classification_dataset(classifcation_data, false);

    auto tobinet = net(layers)(size3d(3, 32, 32));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();
    train(tobinet, classifcation_data.training_data_, 2000, 0.001f, 0.1f, 50);
    //test(tobinet, classifcation_data.training_data_);
    test(tobinet, classifcation_data.test_data_);
    std::cout << frame_string("tobinet elu(1) gentle_max_pool(2, 0.7)") << std::endl;
}

fd::float_t relative_error(fd::float_t x, fd::float_t y)
{
    const auto divisor = fplus::max(std::abs(x), std::abs(y));
    if (divisor < 0.0001f)
        return 0;
    else
        return fplus::abs_diff(x, y) / divisor;
}

bool gradients_equal(fd::float_t max_diff, const fd::float_vec& xs, const fd::float_vec& ys)
{
    assert(xs.size() == ys.size());
    return fplus::all_by(
        fplus::is_less_or_equal_than<fd::float_t>(max_diff),
        fplus::zip_with(relative_error, xs, ys));
}

// http://cs231n.github.io/neural-networks-3/#gradcheck
void gradient_check_backprop_implementation()
{
    std::cout << frame_string("gradient_check_backprop_implementation") << std::endl;

    using namespace fd;

    const auto show_one_value = fplus::show_float_fill_left<fd::float_t>(' ', 7 + 5, 7);
    const auto show_gradient = [show_one_value](const float_vec& xs) -> std::string
    {
        return fplus::show_cont(fplus::transform(show_one_value, xs));
    };

    const auto generate_random_values = [](std::size_t count) -> float_vec
    {
        std::random_device rd; // uses seed from system automatically
        std::mt19937 gen(rd());
        std::normal_distribution<fd::float_t> d(0, 1);
        float_vec values;
        values.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            values.push_back(static_cast<fd::float_t>(d(gen)));
        }
        return values;
    };

    const auto generate_random_data = [&](
        const size3d& in_size,
        const size3d& out_size,
        std::size_t count) -> input_with_output_vec
    {
        input_with_output_vec data;
        data.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto in_vals = generate_random_values(in_size.volume());
            const auto out_vals = generate_random_values(out_size.volume());
            data.push_back({{in_size, in_vals}, {out_size, out_vals}});
        }
        return data;
    };

    const auto test_net_backprop = [&](
        const std::string& name,
        layer_ptr& net,
        std::size_t data_size,
        std::size_t repetitions)
    {
        std::cout << "Testing backprop with " << name << std::endl;
        for (std::size_t i = 0; i < repetitions; ++i)
        {
            const auto data = generate_random_data(
                net->input_size(), net->output_size(), data_size);
            net->random_init_params();
            auto gradient = calc_net_gradient_numeric(net, data);
            auto gradient_backprop = calc_net_gradient_backprop(net, data);
            if (!gradients_equal(0.00001f, gradient_backprop, gradient))
            {
                std::cout << "params            " << show_gradient(net->get_params()) << std::endl;
                std::cout << "gradient          " << show_gradient(gradient) << std::endl;
                std::cout << "gradient_backprop " << show_gradient(gradient_backprop) << std::endl;
                std::cout << "abs diff          " << show_gradient(fplus::zip_with(fplus::abs_diff<fd::float_t>, gradient, gradient_backprop)) << std::endl;
                std::cout << "relative_error    " << show_gradient(fplus::zip_with(relative_error, gradient, gradient_backprop)) << std::endl;
                assert(false);
            }
        }
    };




    auto net_001 = net(
    {
        flatten(),
        fc(2),
        softplus(),
        fc(2),
        identity(),
        fc(2),
        relu(),
        fc(2),
        leaky_relu(0.03f),
        fc(2),
        elu(1),
        fc(2),
        erf(),
        fc(2),
        step(),
        fc(2),
        fast_sigmoid(),
        fc(4),
        sigmoid(),
        fc(3),
        tanh(),
        //softmax()
    })(size3d(1, 1, 2));
    test_net_backprop("net_001", net_001, 10, 10);





    auto net_002 = net(
    {
        conv(size2d(3, 3), 2, 1),
    })(size3d(1, 3, 3));
    test_net_backprop("net_002", net_002, 5, 10);




    auto net_003 = net(
    {
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 5, 5));
    test_net_backprop("conv", net_003, 5, 10);




    auto net_unpool = net(
    {
        conv(size2d(3, 3), 1, 1),
        unpool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 4, 4));
    test_net_backprop("net_unpool", net_unpool, 5, 10);




    auto net_max_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        max_pool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_max_pool", net_max_pool, 5, 10);




    auto net_avg_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        avg_pool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_avg_pool", net_avg_pool, 5, 10);








    auto net_gentle_max_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        gentle_max_pool(2, 1),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_gentle_max_pool", net_gentle_max_pool, 5, 10);




    auto net_softmax = net(
    {
        fc(2),
        softmax(),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_softmax", net_softmax, 1, 10);



    auto net_elu = net(
    {
        fc(2),
        elu(1),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_elu", net_elu, 1, 10);

    auto net_tanh_def = net(
    {
        fc(2),
        tanh(false),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_def", net_tanh_def, 1, 10);

    auto net_tanh_alpha = net(
    {
        fc(2),
        tanh(false, 0.3f),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_alpha", net_tanh_alpha, 1, 10);

    auto net_tanh_lecun = net(
    {
        fc(2),
        tanh(true),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_lecun", net_tanh_lecun, 1, 10);

    auto net_tanh_lecun_alpha = net(
    {
        fc(2),
        tanh(true, 0.2f),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_lecun_alpha", net_tanh_lecun_alpha, 1, 10);



    auto net_006 = net(
    {
        conv(size2d(3, 3), 4, 1), elu(1),
        conv(size2d(1, 1), 2, 1), elu(1),
        max_pool(2),
        flatten(),
        fc(4),
        fc(2),
        //softmax()
    })(size3d(1, 4, 4));
    test_net_backprop("net_006", net_006, 5, 10);




    std::cout << frame_string("Backprop implementation seems to be correct.") << std::endl;
}

int main()
{
    lenna_filter_test();
    gradient_check_backprop_implementation();
    xor_as_net_test();
    gradients_classification_test();
    cifar_10_classification_test();
}
