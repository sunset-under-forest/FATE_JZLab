cmake_minimum_required(VERSION 3.12)
project(FATE_ABY LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # fpic
find_package(ABY QUIET)
if(ABY_FOUND)
    message(STATUS "Found ABY")
elseif (NOT ABY_FOUND AND NOT TARGET ABY::aby)
    message("ABY was not found: add ABY subdirectory")
    add_subdirectory(extern/ABY)
endif()

add_subdirectory(src/millionaire_prob_test)
add_subdirectory(src/test)
add_subdirectory(src/add_and_mul_operator)