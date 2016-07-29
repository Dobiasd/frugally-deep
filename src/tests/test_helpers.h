// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline std::string frame_string(const std::string& str)
{
    return
        std::string(str.size(), '-') + "\n" +
        str + "\n" +
        std::string(str.size(), '-');
}

inline bool file_exists(const std::string& file_path)
{
    return static_cast<bool>(std::ifstream(file_path));
}

inline fd::matrix3d load_col_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

inline fd::matrix3d load_gray_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_gray_img_float_to_matrix3d(img);
}
