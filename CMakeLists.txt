cmake_minimum_required(VERSION 3.16)
project(neat)

set(CMAKE_CXX_STANDARD 17)

add_library(neat include/neat.h src/neat.cpp)

target_include_directories(neat PUBLIC include)