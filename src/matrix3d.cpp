#include "matrix3d.h"

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
