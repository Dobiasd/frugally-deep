#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

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

std::vector<float> img_to_vec(const cv::Mat& img)
{
    std::vector<float> img_vec;
    img_vec.reserve(static_cast<std::size_t>(img.rows * img.cols));
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            img_vec.push_back(img.at<float>(y, x));
        }
    }
    return img_vec;
}

cv::Mat vec_to_img(const std::vector<float>& img_vec, const cv::Size& size)
{
    assert(img_vec.size() == static_cast<std::size_t>(size.width * size.height));
    cv::Mat img(size, CV_32FC1);
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            img.at<float>(y, x) = img_vec[static_cast<std::size_t>(y*img.cols + x)];
        }
    }
    return img;
}

// N-dimensional array.
// Values of the last dimension are stored adjecently in memory.
class tensor
{
public:
    explicit tensor(const std::vector<std::size_t>& dimensions) :
        dimensions_(dimensions),
        dim_idx_factors_(generate_dim_idx_factors_(dimensions_)),
        values_(fplus::product(dimensions_), 0.0f)
    {
    }
    // Values of the last dimension are adjecent in the vector.
    tensor(const std::vector<std::size_t>& dimensions,
        const std::vector<float>& values) :
        dimensions_(dimensions),
        dim_idx_factors_(generate_dim_idx_factors_(dimensions_)),
        values_(values)
    {
        assert(values.size() == fplus::product(dimensions));
    }
    float get(const std::vector<std::size_t>& pos) const
    {
        return values_[idx(pos)];
    }
    void set(const std::vector<std::size_t>& pos, float value)
    {
        values_[idx(pos)] = value;
    }
    std::vector<std::size_t> get_dimensions() const
    {
        return dimensions_;
    }
    // todo
    // tensor get_sub_tensor(std::size_t pos_first_dim)
// todo
//private:
    static std::vector<std::size_t> generate_dim_idx_factors_
        (const std::vector<std::size_t>& dimensions)
    {
        assert(dimensions.size() > 0);
        return
            fplus::tail(
                fplus::scan_right(
                    std::multiplies<std::size_t>(),
                    std::size_t(1),
                    dimensions
                    ));
    }
    std::size_t idx(const std::vector<std::size_t>& pos) const
    {
        assert(pos.size() == dimensions_.size());
        assert(
            fplus::all(
                fplus::zip_with(
                    fplus::is_less<std::size_t>,
                    pos,
                    dimensions_)));
        return
            fplus::sum(
                fplus::zip_with(
                    std::multiplies<std::size_t>(), pos, dim_idx_factors_));
    }
    std::vector<std::size_t> dimensions_;
    std::vector<std::size_t> dim_idx_factors_;
    std::vector<float> values_;
};

std::string show_tensor(const tensor& t)
{
    // todo: recurse through all dimensions-
    std::string result;
    result += "dimensions_";
    result += "\n";
    result += fplus::show_cont(t.dimensions_) + "\n";
    result += "dim_idx_factors_";
    result += "\n";
    result += fplus::show_cont(t.dim_idx_factors_) + "\n";
    result += "values_";
    result += "\n";
    result += fplus::show_cont(t.values_) + "\n";
    return result;
}

class matrix2d
{
public:
    explicit matrix2d(std::size_t height, size_t width) :
        height_(height),
        width_(width),
        values_(height * width, 0.0f)
    {
    }
    float get(std::size_t y, size_t x) const
    {
        return values_[idx(y, x)];
    }
    void set(std::size_t y, size_t x, float value)
    {
        values_[idx(y, x)] = value;
    }
    std::size_t height() const
    {
        return height_;
    }
    std::size_t width() const
    {
        return width_;
    }
private:
    std::size_t idx(std::size_t y, size_t x) const
    {
        return y * width_ + x;
    };
    std::size_t height_;
    std::size_t width_;
    std::vector<float> values_;
};

class size3d
{
public:
    explicit size3d(
        std::size_t depth,
        std::size_t height,
        std::size_t width) :
            depth_(depth),
            height_(height),
            width_(width)
    {
    }
    std::size_t depth() const
    {
        return depth_;
    }
    std::size_t height() const
    {
        return height_;
    }
    std::size_t width() const
    {
        return width_;
    }
    std::size_t area() const
    {
        return depth() * height() * width();
    }
private:
    std::size_t depth_;
    std::size_t height_;
    std::size_t width_;
};

bool operator == (const size3d& lhs, const size3d& rhs)
{
    return
        lhs.depth() == rhs.depth() &&
        lhs.height() == rhs.height() &&
        lhs.width() == rhs.width();
}

class matrix3d
{
public:
    explicit matrix3d(const size3d& size) :
        size_(size),
        values_(size.area(), 0.0f)
    {
    }
    float get(std::size_t z, std::size_t y, size_t x) const
    {
        return values_[idx(z, y, x)];
    }
    void set(std::size_t z, std::size_t y, size_t x, float value)
    {
        values_[idx(z, y, x)] = value;
    }
    const size3d& size() const
    {
        return size_;
    }
private:
    std::size_t idx(std::size_t z, std::size_t y, size_t x) const
    {
        return z * size().height() * size().width() + y * size().width() + x;
    };
    size3d size_;
    std::vector<float> values_;
};

std::string show_matrix3d(const matrix3d& m)
{
    std::string str;
    str += "[";
    for (std::size_t z = 0; z < m.size().depth(); ++z)
    {
        str += "[";
        for (std::size_t y = 0; y < m.size().height(); ++y)
        {
            for (std::size_t x = 0; x < m.size().width(); ++x)
            {
                str += std::to_string(m.get(z, y, x)) + ",";
            }
            str += "]\n";
        }
        str += "]\n";
    }
    str += "]";
    return str;
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

class layer
{
public:
    virtual ~layer()
    {
    }
    virtual matrix3d forward_pass(const matrix3d& input) const = 0;
    virtual std::size_t param_count() const = 0;
    virtual std::vector<float> get_params() const = 0;
    virtual void set_params(const std::vector<float>& params) = 0;
    virtual std::size_t input_depth() const = 0;
    virtual std::size_t output_depth() const = 0;
};

typedef std::shared_ptr<layer> layer_ptr;

class filter
{
public:
    explicit filter(const matrix3d& m) : m_(m)
    {
    }
    std::size_t param_count() const
    {
        return m_.size().area();
    }
    std::vector<float> get_params() const
    {
        std::vector<float> params;
        params.reserve(param_count());
        for (std::size_t z = 0; z < m_.size().depth(); ++z)
        {
            for (std::size_t y = 0; y < m_.size().height(); ++y)
            {
                for (std::size_t x = 0; x < m_.size().width(); ++x)
                {
                    params.push_back(m_.get(z, y, x));
                }
            }
        }
        return params;
    }
    void set_params(const std::vector<float>& params)
    {
        assert(params.size() == param_count());
        std::size_t i = 0;
        for (std::size_t z = 0; z < m_.size().depth(); ++z)
        {
            for (std::size_t y = 0; y < m_.size().height(); ++y)
            {
                for (std::size_t x = 0; x < m_.size().width(); ++x)
                {
                    m_.set(z, y, x, params[++i]);
                }
            }
        }
    }
    const size3d& size() const
    {
        return m_.size();
    }
    matrix3d get_matrix3d() const
    {
        return m_;
    }
    float get(std::size_t z, std::size_t y, size_t x) const
    {
        return m_.get(z, y, x);
    }
private:
    matrix3d m_;
};

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

matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol)
{
    // todo: padding
    assert(in_vol.size().depth() == filters.size());
    matrix3d out_vol(size3d(
        filters.size(),
        in_vol.size().height(),
        in_vol.size().width()));
    // todo: use im_to_col and matrix multiplication for performance (?)
    for (std::size_t k = 0; k < filters.size(); ++k)
    {
        for (std::size_t y = 1; y < in_vol.size().height() - 1; ++y)
        {
            for (std::size_t x = 1; x < in_vol.size().width() - 1; ++x)
            {
                float val = 0.0f;
                for (std::size_t z = 0; z < filters[k].size().depth(); ++z)
                {
                    for (std::size_t yf = 0; yf < filters[k].size().height(); ++yf)
                    {
                        for (std::size_t xf = 0; xf < filters[k].size().width(); ++xf)
                        {
                            val += filters[k].get(z, yf, xf) *
                                in_vol.get(z, y - 1 + yf, x - 1 + xf);
                        }
                    }
                }
                out_vol.set(k, y, x, val);
            }
        }
    }
    return out_vol;
}

class conv_layer : public layer
{
public:
    explicit conv_layer(const std::vector<filter>& filters) :
        filters_(filters)
    {
        assert(fplus::is_not_empty(filters));
        auto filter_sizes =
            fplus::transform([](const filter& f) { return f.size(); },
            filters_);
        assert(fplus::all_the_same(filter_sizes));
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return convolve(filters_, input);
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const filter& f) { return f.param_count(); },
            filters_);
        return fplus::sum(counts);
    }
    std::vector<float> get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const filter& f) { return f.get_params(); },
                filters_));
    }
    void set_params(const std::vector<float>& params) override
    {
        assert(params.size() == param_count());
        auto params_per_filter =
            fplus::split_every(filters_.front().param_count(), params);
        for (std::size_t i = 0; i < filters_.size(); ++i)
        {
            filters_[i].set_params(params_per_filter[i]);
        }
    }
    std::size_t input_depth() const override
    {
        return filters_.front().get_matrix3d().size().depth();
    }
    std::size_t output_depth() const override
    {
        return filters_.size();
    }
private:
    std::vector<filter> filters_;
};


class conv_net : public layer
{
public:
    explicit conv_net(const std::vector<layer_ptr>& layer_ptrs) :
        layer_ptrs_(layer_ptrs)
    {
        assert(is_layer_ptr_chain_valid(layer_ptrs_));
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        auto current_volume = input;
        for (const auto& layer_ptr : layer_ptrs_)
        {
            current_volume = layer_ptr->forward_pass(current_volume);
        }
        return current_volume;
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const layer_ptr& l) { return l->param_count(); },
            layer_ptrs_);
        return fplus::sum(counts);
    }
    std::vector<float> get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const layer_ptr& l) { return l->get_params(); },
                layer_ptrs_));
    }
    void set_params(const std::vector<float>& params) override
    {
        auto layer_param_counts = fplus::transform(
            [](const layer_ptr& l) { return l->param_count(); },
            layer_ptrs_);
        auto split_idxs =
            fplus::scan_left_1(std::plus<std::size_t>(), layer_param_counts);
        auto params_per_layer = fplus::split_at_idxs(split_idxs, params);
        for (std::size_t i = 0; i < layer_ptrs_.size(); ++i)
        {
            layer_ptrs_[i]->set_params(params_per_layer[i]);
        }
    }
    std::size_t input_depth() const override
    {
        return layer_ptrs_.front()->input_depth();
    }
    std::size_t output_depth() const override
    {
        return layer_ptrs_.back()->output_depth();
    }
private:
    static bool is_layer_transition_valid(
        const std::pair<layer_ptr, layer_ptr>& layer_ptr_pair)
    {
        return
            layer_ptr_pair.first->output_depth() ==
            layer_ptr_pair.second->input_depth();
    }
    static bool is_layer_ptr_chain_valid(
        const std::vector<layer_ptr>& layer_ptrs)
    {
        // todo: what about non-conv-layers like FC?
        return
            fplus::all_by(
                [](const layer_ptr& ptr) -> bool
                {
                    return static_cast<bool>(ptr);
                },
                layer_ptrs) &&
            fplus::all_by(
                is_layer_transition_valid,
                fplus::overlapping_pairs(layer_ptrs));
    }
    std::vector<layer_ptr> layer_ptrs_;
};

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
// Conv transpose layer
// Skip conn aka comp graph
// was ist ein softmax-layer nochmal?
// Affine layer- flow layer?

// zweites video dabei, was die differenzframes drin hat

// anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video

// oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf