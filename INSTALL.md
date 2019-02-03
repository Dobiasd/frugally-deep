frugally-deep
=============

Installation
------------

You can install frugally-deep using cmake as shown below, or (if you prefer) download the [code](https://github.com/Dobiasd/frugally-deep/archive/master.zip) (and the [code](https://github.com/Dobiasd/FunctionalPlus/archive/master.zip) of [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)), extract it and tell your compiler to use the `include` directories.

```
git clone https://github.com/Dobiasd/FunctionalPlus
cd FunctionalPlus
mkdir -p build && cd build
cmake ..
make && sudo make install
cd ../..

sudo apt install mercurial
hg clone https://bitbucket.org/eigen/eigen/
cd eigen
mkdir -p build && cd build
cmake ..
make && sudo make install
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
cd ../..

git clone https://github.com/nlohmann/json
cd json
git checkout v3.1.2
mkdir -p build && cd build
cmake ..
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
git clone https://github.com/onqtam/doctest.git
cd doctest
git checkout tags/1.2.9
mkdir -p build && cd build
cmake -DDOCTEST_WITH_TESTS=OFF -DDOCTEST_WITH_MAIN_IN_STATIC_LIB=OFF ..
make && sudo make install
cd ../..

# build unit tests
cd frugally-deep
mkdir -p build && cd build
cmake -DFDEEP_BUILD_UNITTEST=ON ..
make unittest
cd ../..
```

Or if you would like to test exhaustively (and have plenty of time):
```
cmake -DFDEEP_BUILD_UNITTEST=ON -DFDEEP_BUILD_FULL_TEST=ON ..
make unittest
```


### Installation using [Conan C/C++ package manager](https://conan.io)

Just add a *conanfile.txt* with frugally-deep as a requirement and chose the generator for your project.

```
[requires]
frugally-deep/v0.7.6-p0@dobiasd/stable

[generators]
cmake
```

Then install it:

```
$ conan install conanfile.txt
```

### Installation using the [Hunter CMake package manager](https://github.com/ruslo/hunter)

The [First Step](https://docs.hunter.sh/en/latest/quick-start/boost-components.html#first-step) section of the [Hunter documentation](https://docs.hunter.sh/en/latest/index.html) shows how to get started.  The following is reproduced from the the [frugally-deep](https://docs.hunter.sh/en/latest/packages/pkg/frugally-deep.html?highlight=frugally-deep) package notes.

First, add the `HunterGate` module to your project, i.e.:

```
mkdir -p cmake
wget https://raw.githubusercontent.com/hunter-packages/gate/master/cmake/HunterGate.cmake -O cmake/HunterGate.cmake
```

You can then integrate `frugally-deep` (and other packages) based on the following `CMakeLists.txt` example:

```
cmake_minimum_required(VERSION 3.0) # minimum requirement for Hunter

include("cmake/HunterGate.cmake") # teach your project about Hunter (before project())
HunterGate( # Latest release shown here: https://github.com/ruslo/hunter/releases
    URL "https://github.com/ruslo/hunter/archive/v0.20.17.tar.gz"
    SHA1 "d7d1d5446bbf20b78fa5ac1b52ecb67a01c3790e"
)

project(sample-frugally-deep)

hunter_add_package(frugally-deep)
find_package(frugally-deep CONFIG REQUIRED)

add_executable(foo foo.cpp)
target_link_libraries(foo PUBLIC frugally-deep::fdeep) # add frugally-deep and dependencies (libs/includes/flags/definitions)
```
