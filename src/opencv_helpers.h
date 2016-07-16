// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/conv_layer.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/matrix3d.h"
#include "frugally_deep/multi_layer_net.h"
#include "frugally_deep/avg_pool_layer.h"
#include "frugally_deep/max_pool_layer.h"
#include "frugally_deep/unpool_layer.h"

#include <opencv2/opencv.hpp>

#include <cassert>

inline fd::matrix3d cv_bgr_img_float_to_matrix3d(const cv::Mat& img)
{
    assert(img.type() == CV_32FC3);
    fd::matrix3d m(fd::size3d(
        3,
        static_cast<std::size_t>(img.cols),
        static_cast<std::size_t>(img.rows)));
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3f col =
                img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < 3; ++c)
            {
                m.set(c, y, x,
                    static_cast<float>(col[static_cast<int>(c)]));
            }
        }
    }
    return m;
}

inline fd::matrix3d cv_float_kernel_to_matrix3d(const cv::Mat& kernel,
    std::size_t depth, std::size_t z)
{
    assert(kernel.type() == CV_32FC1);
    fd::matrix3d m(fd::size3d(
        depth,
        static_cast<std::size_t>(kernel.cols),
        static_cast<std::size_t>(kernel.rows)));
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            float val =
                kernel.at<float>(static_cast<int>(y), static_cast<int>(x));
            m.set(z, y, x, val);
        }
    }
    return m;
}

inline cv::Mat matrix3d_to_cv_bgr_img_float(const fd::matrix3d& m)
{
    assert(m.size().depth() == 3);
    cv::Mat img(
        static_cast<int>(m.size().height()),
        static_cast<int>(m.size().width()),
        CV_32FC3);
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3f col(m.get(0, y, x), m.get(1, y, x), m.get(2, y, x));
            img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x)) = col;
        }
    }
    return img;
}

inline fd::layer_ptr cv_kernel_to_layer(const cv::Mat& cv_kernel)
{
    std::vector<fd::filter> filters = {
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 0)),
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 1)),
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 2))
    };
    return std::make_shared<fd::conv_layer>(filters);
}

inline cv::Mat filter2D_via_net(const cv::Mat& img, const cv::Mat& cv_kernel)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    std::vector<fd::layer_ptr> layers = {cv_kernel_to_layer(cv_kernel)};
    fd::multi_layer_net net(layers);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat filter2Ds_via_net(const cv::Mat& img,
    const std::vector<cv::Mat>& cv_kernels)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    auto layers = fplus::transform(cv_kernel_to_layer, cv_kernels);
    fd::multi_layer_net net(layers);
    auto out_vol = net.forward_pass(in_vol);
    //std::cout << fplus::show_cont(net.get_params()) << std::endl;
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat shrink_via_net(const cv::Mat& img, std::size_t scale_factor)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    fd::avg_pool_layer net(scale_factor);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat grow_via_net(const cv::Mat& img, std::size_t scale_factor)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    fd::unpool_layer net(scale_factor);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat uchar_img_to_float_img(const cv::Mat& uchar_img)
{
    assert(uchar_img.type() == CV_8UC3);
    cv::Mat result;
    uchar_img.convertTo(result, CV_32FC3, 1.0f/255.0f);
    return result;
}

inline cv::Mat float_ing_to_uchar_img(const cv::Mat& floatImg)
{
    assert(floatImg.type() == CV_32FC3);
    cv::Mat result;
    floatImg.convertTo(result, CV_8UC3, 255.0f);
    return result;
}

inline cv::Mat normalize_float_img(const cv::Mat& floatImg)
{
    assert(floatImg.type() == CV_32FC3);
    cv::Mat result;
    cv::normalize(floatImg, result, 0, 1, cv::NORM_MINMAX);
    return result;
}
