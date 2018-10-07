#include <fdeep/fdeep.hpp>
#include <iostream>

int main()
{
    const fdeep::tensor3 t(fdeep::shape_hwc(1, 1, 4), {1, 2, 3, 4});
    std::cout << fdeep::show_tensor3(t) << std::endl;
}