#include <fdeep/fdeep.hpp>
#include <iostream>

int main()
{
    const fdeep::tensor5 t(fdeep::shape5(4), {1, 2, 3, 4});
    std::cout << fdeep::show_tensor5(t) << std::endl;
}