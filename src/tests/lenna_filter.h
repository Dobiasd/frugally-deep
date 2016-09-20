// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "test_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.hpp>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline void lenna_filter_test()
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
