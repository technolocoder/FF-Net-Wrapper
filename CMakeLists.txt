cmake_minimum_required(VERSION 3.2...3.8)
project(FF_Net VERSION 1.0 DESCRIPTION "Test FF" LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_subdirectory(Test)
add_subdirectory(FF)