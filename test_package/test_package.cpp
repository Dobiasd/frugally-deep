#include <fdeep/fdeep.hpp>
#include <iostream>

int main()
{
    const fdeep::tensor3 t(fdeep::shape3(4, 1, 1), {1, 2, 3, 4});
    std::cout << fdeep::show_tensor3(t) << std::endl;
}