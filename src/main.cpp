// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

// http://stackoverflow.com/a/21802936
std::vector<unsigned char> read_file(const std::string& filename)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    std::vector<unsigned char> vec;
    vec.reserve(static_cast<std::size_t>(fileSize));

    // read the data:
    vec.insert(vec.begin(),
               std::istream_iterator<unsigned char>(file),
               std::istream_iterator<unsigned char>());

    return vec;
}

fd::input_with_output parse_cifar_10_bin_line(
    const std::vector<unsigned char>& vec)
{
    assert(vec.size() == 3073);
    fd::matrix3d output(fd::size3d(1, 1, 10));
    output.set(0, 0, vec[0], 1);
    fd::matrix3d input(fd::size3d(3, 32, 32));
    std::size_t vec_i = 0;
    for (std::size_t z = 0; z < input.size().depth(); ++z)
    {
        for (std::size_t y = 0; y < input.size().height(); ++y)
        {
            for (std::size_t x = 0; x < input.size().depth(); ++x)
            {
                input.set(z, y, x, vec[++vec_i]);
            }
        }
    }
    return {input, output};
}

fd::input_with_output_vec load_cifar_10_bin_file(const std::string& file_path)
{
    const auto bytes = read_file(file_path);
    assert(bytes.size() == 30730000);
    const auto lines = fplus::split_every(3073, bytes);
    assert(lines.size() == 10000);
    return fplus::transform(parse_cifar_10_bin_line, lines);
}

fd::input_with_output_vec load_cifar_10_bin_training()
{
    return fplus::concat(std::vector<fd::input_with_output_vec>({
        load_cifar_10_bin_file("data_batch_1.bin"),
        load_cifar_10_bin_file("data_batch_2.bin"),
        load_cifar_10_bin_file("data_batch_3.bin"),
        load_cifar_10_bin_file("data_batch_4.bin"),
        load_cifar_10_bin_file("data_batch_5.bin")}));
}

fd::input_with_output_vec load_cifar_10_bin_test()
{
    return load_cifar_10_bin_file("test_batch.bin");
}

int main()
{
    auto test_pairs = load_cifar_10_bin_test();

    cv::Mat img_uchar = cv::imread("images/lenna_512x512.png", cv::IMREAD_COLOR);
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
    filtered1 = float_ing_to_uchar_img(filtered1);
    cv::imwrite("lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2Ds_via_net(img, {kernel_scharr_x, kernel_scharr_x});
    filtered2 = shrink_via_net(filtered2, 2);
    filtered2 = grow_via_net(filtered2, 2);
    filtered2 = normalize_float_img(filtered2);
    filtered2 = float_ing_to_uchar_img(filtered2);
    cv::imwrite("lenna_512x512_filtered2.png", filtered2);
}
