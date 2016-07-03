#include "conv_net.h"
#include "conv_layer.h"
#include "filter.h"
#include "matrix3d.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <iostream>
#include <random>

cv::Mat filter2DNestedForLoops(const cv::Mat& img, const cv::Mat kernel)
{
    cv::Mat dst = img.clone();
    for (int y = 1; y < img.rows - 1; ++y)
    {
        for (int x = 1; x < img.cols - 1; ++x)
        {
            float acc = 0.0f;
            for (int yk = 0; yk < kernel.rows; ++yk)
            {
                for (int xk = 0; xk < kernel.cols; ++xk)
                {
                    acc +=
                        img.at<float>(y + yk - 1, x + xk - 1) *
                        kernel.at<float>(yk, xk);
                }
            }
            dst.at<float>(y, x) = acc;
        }
    }
    return dst;
}


matrix3d cv_bgr_img_to_matrix3d(const cv::Mat& img)
{
    assert(img.type() == CV_8UC3);
    matrix3d m(size3d(
        3,
        static_cast<std::size_t>(img.cols),
        static_cast<std::size_t>(img.rows)));
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3b col =
                img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < 3; ++c)
            {
                m.set(c, y, x,
                    static_cast<float>(col[static_cast<int>(c)]) / 256.0f);
            }
        }
    }
    return m;
}

matrix3d cv_float_kernel_to_matrix3d(const cv::Mat& kernel,
    std::size_t depth, std::size_t z)
{
    assert(kernel.type() == CV_32FC1);
    matrix3d m(size3d(
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

cv::Mat matrix3d_to_cv_bgr_img(const matrix3d& m)
{
    assert(m.size().depth() == 3);
    cv::Mat img(
        static_cast<int>(m.size().height()),
        static_cast<int>(m.size().width()),
        CV_8UC3);
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3b col(
                static_cast<unsigned char>(std::max(0.0f, m.get(0, y, x) * 256)),
                static_cast<unsigned char>(std::max(0.0f, m.get(1, y, x) * 256)),
                static_cast<unsigned char>(std::max(0.0f, m.get(2, y, x) * 256))
                );
            img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x)) = col;
        }
    }
    return img;
}


filter random_filter(const size3d& size)
{
    /*
    std::random_device rd;
    std::mt19937 mt(rd());
    std::size_t parameter_count = size.area();
    float multiplicator = 1.0f / static_cast<float>(parameter_count);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    */
    // todo befuellen
    return filter(matrix3d(size));
}



cv::Mat filter2DMatMult(const cv::Mat& img, const cv::Mat& cv_kernel)
{
    //auto img_vec = img_to_vec(img);
    //auto kernel_vec = img_to_vec(kernel);
    //return vec_to_img(img_vec, img.size());
    matrix3d in_vol = cv_bgr_img_to_matrix3d(img);
    std::vector<filter> filters = {
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 0)),
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 1)),
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 2))
    };

    std::cout << "asdasd " << show_matrix3d(filters[0].get_matrix3d()) << std::endl;
    std::cout << "asdasd " << show_matrix3d(filters[1].get_matrix3d()) << std::endl;
    std::cout << "asdasd " << show_matrix3d(filters[2].get_matrix3d()) << std::endl;

    std::vector<layer_ptr> layers = {std::make_shared<conv_layer>(filters)};
    conv_net net = conv_net(layers);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img(out_vol);
    return result;
}

int main()
{
    cv::Mat img = cv::imread("images/lenna_512x512.png", cv::IMREAD_COLOR);

    cv::Mat kernel(cv::Size(3,3), CV_32FC1, cv::Scalar(0));
    kernel.at<float>(0,0) =   3.0f / 32.0f;
    kernel.at<float>(1,0) =  10.0f / 32.0f;
    kernel.at<float>(2,0) =   3.0f / 32.0f;
    kernel.at<float>(0,1) =   0.0f / 32.0f;
    kernel.at<float>(1,1) =   0.0f / 32.0f;
    kernel.at<float>(2,1) =   0.0f / 32.0f;
    kernel.at<float>(0,2) =  -3.0f / 32.0f;
    kernel.at<float>(1,2) = -10.0f / 32.0f;
    kernel.at<float>(2,2) =  -3.0f / 32.0f;

    cv::Mat filtered1;
    filter2D(img, filtered1, -1, kernel);
    cv::imwrite("lenna_512x512_filtered1.png", filtered1);

    //cv::Mat filtered2 = filter2DNestedForLoops(img, kernel);
    //cv::imwrite("lenna_512x512_filtered2.png", filtered2);

    cv::Mat filtered3 = filter2DMatMult(img, kernel);
    cv::imwrite("lenna_512x512_filtered3.png", filtered3);

/*
    auto error_func = [&]
    {
        net_calculate
    };
    net.set_
*/




    /*
    tensor t1({4,7,12,3});
    t1.set({3,1,9,2}, 42.0f);
    std::cout << t1.get({3,1,9,2}) << std::endl;

    tensor t2({3,3,3});
    t2.set({0,0,0}, 1.0f);
    t2.set({1,1,1}, 5.0f);
    std::cout << show_tensor(t2) << std::endl;
    */
}


// todo:
// Conv free func
// transposed convolution layer
// Skip conn aka comp graph
// was ist ein softmax-layer nochmal?
// Affine layer- flow layer?

// zweites video dabei, was die differenzframes drin hat

// anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video

// oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf