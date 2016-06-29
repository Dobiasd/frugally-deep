#!/usr/bin/env bash
rm -r build
mkdir build
clang++-3.6 -std=c++11 -O3 -Wall -Wextra -pedantic -Wshadow -Werror -Weffc++ -Wconversion -Wsign-conversion -Wctor-dtor-privacy -Wreorder -Wold-style-cast -pthread src/DeepVideoCompression.cpp `pkg-config --cflags --libs opencv` -o build/DeepVideoCompression