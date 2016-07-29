#pragma once

#include <fstream>

// http://stackoverflow.com/a/10409376/1866775
inline std::int32_t reverse_int_bytes(std::int32_t i)
{
    std::uint8_t c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return
        (static_cast<std::int32_t>(c1) << 24) +
        (static_cast<std::int32_t>(c2) << 16) +
        (static_cast<std::int32_t>(c3) << 8) +
        static_cast<std::int32_t>(c4);
}

inline std::int32_t concat_bytes_to_int(const std::vector<std::uint8_t>& bytes)
{
    assert(bytes.size() == 4);
    std::int32_t i =
        (bytes[3] << 24) +
        (bytes[2] << 16) +
        (bytes[1] << 8) +
        (bytes[0]);
    return i;
}

inline std::vector<std::uint8_t> read_mnist_label_file(
    const std::string& file_path)
{
    const auto bytes = fplus::read_binary_file(file_path)();
	fplus::write_binary_file( file_path + ".asd", bytes )();

    const auto header_and_body = fplus::split_at_idx(8, bytes);

    std::int32_t magic_number =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(0, 4, header_and_body.first)));
    assert(magic_number == 2049);

    std::int32_t number_of_items =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(4, 8, header_and_body.first)));

    assert(number_of_items == 60000 || number_of_items == 10000);
    assert(header_and_body.second.size() ==
        static_cast<std::size_t>(number_of_items));

    // todo remove
    //return fplus::take(28*28*20, header_and_body.second);
    return header_and_body.second;
}

inline std::vector<std::uint8_t> read_mnist_image_file(
    const std::string& file_path)
{
    const auto bytes = fplus::read_binary_file(file_path)();

    const auto header_and_body = fplus::split_at_idx(16, bytes);

    std::int32_t magic_number =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(0, 4, header_and_body.first)));
    assert(magic_number == 2051);

    std::int32_t number_of_items =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(4, 8, header_and_body.first)));

    assert(number_of_items == 60000 || number_of_items == 10000);
    assert(header_and_body.second.size() ==
        static_cast<std::size_t>(number_of_items) * 28*28);

    std::int32_t number_of_rows =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(8, 12, header_and_body.first)));
    assert(number_of_rows == 28);

    std::int32_t number_of_columns =
        reverse_int_bytes(
            concat_bytes_to_int(
                fplus::get_range(8, 12, header_and_body.first)));
    assert(number_of_columns == 28);

    // todo remove
    //return fplus::take(28*28*20, header_and_body.second);
    return header_and_body.second;
}

inline std::vector<fd::matrix3d> mnist_image_data_to_matrix3ds(
    const std::vector<std::uint8_t>& vec)
{
    assert(vec.size() % (28*28) == 0);
    std::vector<fd::matrix3d> result;
    const auto splits = fplus::split_every(28*28, vec);
    return fplus::transform(
        [](const std::vector<std::uint8_t>& data) -> fd::matrix3d
    {
        return fd::matrix3d(fd::size3d(1, 28, 28),
            fplus::convert_container_and_elems<fd::float_vec>(data));
    }, splits);
}

inline std::vector<fd::matrix3d> mnist_label_data_to_matrix3ds(
    const std::vector<std::uint8_t>& vec)
{
    std::vector<fd::matrix3d> result;
    return fplus::transform(
        [](std::uint8_t label) -> fd::matrix3d
    {
        fd::matrix3d m(fd::size3d(1, 10, 1));
        m.set(0, static_cast<std::size_t>(label), 0, 1);
        return m;
    }, vec);
}

inline fd::classification_dataset read_mnist(const std::string& base_dir)
{
    const auto test_labels = mnist_label_data_to_matrix3ds(
        read_mnist_label_file(base_dir + "/t10k-labels.idx1-ubyte"));
    const auto test_images = mnist_image_data_to_matrix3ds(
        read_mnist_image_file(base_dir + "/t10k-images.idx3-ubyte"));
    const auto training_labels = mnist_label_data_to_matrix3ds(
        read_mnist_label_file(base_dir + "/train-labels.idx1-ubyte"));
    const auto training_images = mnist_image_data_to_matrix3ds(
        read_mnist_image_file(base_dir + "/train-images.idx3-ubyte"));
    const auto create_input_with_output = [](
            const fd::matrix3d& input, const fd::matrix3d& output)
                -> fd::input_with_output
        {
            return {input, output};
        };
    return {
        fplus::zip_with(
            create_input_with_output, training_images, training_labels),
        fplus::zip_with(
            create_input_with_output, test_images, test_labels)
    };
}
