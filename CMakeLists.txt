cmake_minimum_required(VERSION 3.16)
project(neat)

set(CMAKE_CXX_STANDARD 17)
set(NEAT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(GA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/genetic-algorithm/include)
set(NETWORK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/network/include)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

enable_testing()

add_library(neat include/neat.h include/neat_genomes.h include/gene_pool.h
        src/neat.cpp src/neat_genomes.cpp src/gene_pool.cpp)

target_include_directories(neat PUBLIC include ${GA_INCLUDE_DIR} ${NETWORK_INCLUDE_DIR})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(NEAT_test)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(genetic-algorithm)
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(network)
endif()