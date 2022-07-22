frugally-deep
=============

Installation
------------

You can install frugally-deep using cmake as shown below, or (if you prefer) download the [code](https://github.com/Dobiasd/frugally-deep/archive/master.zip) (and the [code](https://github.com/Dobiasd/FunctionalPlus/archive/master.zip) of [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)), extract it and tell your compiler to use the `include` directories.

```
git clone -b 'v0.2.18-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus
cd FunctionalPlus
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

git clone -b '3.4.0' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir -p build && cd build
cmake ..
make && sudo make install
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
cd ../..

git clone -b 'v3.10.5' --single-branch --depth 1 https://github.com/nlohmann/json
cd json
mkdir -p build && cd build
cmake -DJSON_BuildTests=OFF ..
make && sudo make install
cd ../..

git clone https://github.com/Dobiasd/frugally-deep
cd frugally-deep
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..
```

Building the tests (optional) requires [doctest](https://github.com/onqtam/doctest). Unit Tests are disabled by default – they are enabled and executed by:

```
# install doctest
git clone -b '2.3.5' --single-branch --depth 1 https://github.com/onqtam/doctest.git
cd doctest
mkdir -p build && cd build
cmake .. -DDOCTEST_WITH_TESTS=OFF -DDOCTEST_WITH_MAIN_IN_STATIC_LIB=OFF
make && sudo make install
cd ../..

# build unit tests
cd frugally-deep
mkdir -p build && cd build
cmake -DFDEEP_BUILD_UNITTEST=ON ..
make unittest
cd ../..
```


### Installation using [Conan C/C++ package manager](https://conan.io)

Just add a *conanfile.txt* with frugally-deep as a requirement and chose the generator for your project.

```
[requires]
frugally-deep/v0.15.19-p0@dobiasd/stable

[generators]
cmake
```

Then install it:

```
$ conan install conanfile.txt
```

### Installation using the [Hunter CMake package manager](https://github.com/ruslo/hunter)
The [First Step](https://docs.hunter.sh/en/latest/quick-start/boost-components.html#first-step) section of the [Hunter documentation](https://docs.hunter.sh/en/latest/index.html) shows how to get started.

Since the version of the package on hunter is out of date, the procedure below covers installation using a locally hosted version of the repo (through submodules). A sample project using this to run VGG16 is available at https://github.com/kmader/fd_demo

The basic idea is to use the standard hunter setup but to add a git submodule to your repository containing frugally-deep. Hunter will then use the code in that submodule to build the library (https://docs.hunter.sh/en/latest/user-guides/hunter-user/git-submodule.html?highlight=GIT_SUBMODULE).

Your CMakeLists.txt should look something like

```cmake
cmake_minimum_required(VERSION 3.0) # minimum requirement for Hunter

include("cmake/HunterGate.cmake") # teach your project about Hunter (before project())
HunterGate( # Latest release shown here: https://github.com/ruslo/hunter/releases
    URL "https://github.com/ruslo/hunter/archive/v0.20.17.tar.gz"
    SHA1 "d7d1d5446bbf20b78fa5ac1b52ecb67a01c3790e"
    LOCAL # <----- load cmake/Hunter/config.cmake
)

project(sample-frugally-deep)

hunter_add_package(frugally-deep)
find_package(frugally-deep CONFIG REQUIRED)

add_executable(foo foo.cpp)
target_link_libraries(foo PUBLIC frugally-deep::fdeep) # add frugally-deep and dependencies (libs/includes/flags/definitions)
```

You will then need to create a `cmake/` directory with the HunterGate script in it

```bash
mkdir -p cmake
wget https://raw.githubusercontent.com/hunter-packages/gate/master/cmake/HunterGate.cmake -O cmake/HunterGate.cmake
```

Finally you will need a `Hunter/config.cmake` to link to the submodule

```bash
mkdir -p cmake/Hunter
echo 'hunter_config(frugally-deep GIT_SUBMODULE "lib/frugally-deep")' > cmake/Hunter/config.cmake
```

### Installation using [vcpkg](https://github.com/microsoft/vcpkg)
See [Getting Started](https://github.com/microsoft/vcpkg#getting-started) to get vcpkg up and running.
The only step after the installation of vcpkg is to install frugally-deep with 
```bash
vcpkg install frugally-deep
```
If you need double precision, install frugally-deep with
```bash
vcpkg install frugally-deep[double]
```

Then add the following lines to your CMakeFiles.txt: 
```cmake
find_package(frugally-deep CONFIG REQUIRED)
target_link_libraries(main PRIVATE frugally-deep::fdeep)
```
