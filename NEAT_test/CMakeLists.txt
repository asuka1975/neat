cmake_minimum_required(VERSION 3.16)
project(NEAT_test)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)

add_executable(NEAT_test_run test.cpp)

target_link_libraries(NEAT_test_run neat genetic network gtest gmock -pthread)

target_include_directories(NEAT_test_run PUBLIC
        ${NEAT_INCLUDE_DIR} ${GA_INCLUDE_DIR} ${NETWORK_INCLUDE_DIR})